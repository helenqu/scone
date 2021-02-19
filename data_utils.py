import os
import numpy as np
import tensorflow as tf

# retrieves heatmap data, associated id, and label from TFRecord data files
# requires:
#   - raw_record
#   - INPUT_SHAPE
#   - CATEGORICAL
def get_images(raw_record, input_shape, categorical): # ASSUMES has ids always
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(raw_record, image_feature_description)
    image = tf.reshape(tf.io.decode_raw(example['image_raw'], tf.float64), input_shape)
    image = image / tf.reduce_max(image[:,:,0])
    if not categorical:
        label = 1 if example['label'] == 0 else 0 # Ia's are labeled as 0 for categorical but change to 1 for binary

    return image, label, example['id']

# balances classes, splits dataset into train/validation/test sets
# requires:
#   - dataset
#   - train_proportion
#   - val_proportion
def stratified_split(dataset, train_proportion, val_proportion): # ASSUMES has ids always
    by_type_data_lists = {}

    for i, elem in enumerate(dataset):
        sn_type = elem[1].numpy()
        heatmap = elem[0].numpy()

        # construct mapping from type to all heatmaps/ids of that type
        if sn_type in by_type_data_lists:
            by_type_data_lists[sn_type] = np.append(by_type_data_lists[sn_type], [[elem[2].numpy(), heatmap]], axis=0)
        else:
            by_type_data_lists[sn_type] = np.array([[elem[2].numpy(), heatmap]])
    # mapping from type to number of heatmaps of that type
    by_type_data_lengths = {k: len(v) for k,v in by_type_data_lists.items()}

    # if classes not balanced, balance them 
    if np.amax(list(by_type_data_lengths.values())) != np.amin(list(by_type_data_lengths.values())):
        generator = np.random.default_rng()
        num_to_keep = np.amin(list(by_type_data_lengths.values()))
        print("classes not balanced: max num {}, min num {}".format(np.amax(list(by_type_data_lengths.values())), num_to_keep))
        for sn_type, sn_data in by_type_data_lists.items():
            by_type_data_lists[sn_type] = generator.choice(sn_data, num_to_keep, replace=False)
            print("type {} has {} examples".format(sn_type, len(by_type_data_lists[sn_type])))

    train_set = None
    val_set = None
    test_set = None
    # make a TF dataset from all heatmaps/ids of each type
    for sntype, data in by_type_data_lists.items():
        dataset = tf.data.Dataset.from_tensor_slices((tf.stack(data[:,1], axis=0), [sntype]*len(data), list(data[:,0].astype(np.int32))))
        dataset = dataset.shuffle(len(data))

        # take from each with correct proportion to make stratified split train/val/test
        if train_set != None:
            train_set = train_set.concatenate(dataset.take(int(len(data)*train_proportion)))
            test_val_set = dataset.skip(int(len(data)*train_proportion))
            val_set = val_set.concatenate(test_val_set.take(int(len(data)*val_proportion)))
            test_set = test_set.concatenate(test_val_set.skip(int(len(data)*val_proportion)))
        else:
            train_set = dataset.take(int(len(data)*train_proportion))
            test_val_set = dataset.skip(int(len(data)*train_proportion))
            val_set = test_val_set.take(int(len(data)*val_proportion))
            test_set = test_val_set.skip(int(len(data)*val_proportion))

    full_dataset_size = len(data)*len(by_type_data_lists.keys) #full dataset size = heatmaps per type * num types
    train_set = train_set.shuffle(full_dataset_size)
    val_set = val_set.shuffle(int(full_dataset_size*val_proportion))
    test_set = test_set.shuffle(int(full_dataset_size*val))

    return train_set, val_set, test_set

# extract ids from all datasets post-caching
# requires:
#   - train_set, val_set, test_set
#   - extract_ids_from_dataset helper function
def extract_ids(train_set, val_set, test_set):
    train_ids, train_set = extract_ids_from_dataset(train_set)
    val_ids, val_set = extract_ids_from_dataset(val_set)
    test_ids, test_set = extract_ids_from_dataset(test_set)
    
    return train_set, val_set, test_set, train_ids, val_ids, test_ids

# helper function for extract_ids
# requires:
#   - cached_dataset
def extract_ids_from_dataset(cached_dataset):
    ids = tf.TensorArray(dtype=tf.int32,
                      size=0,
                      dynamic_size=True)
    dataset = cached_dataset.map(lambda heatmap, label, snid: (heatmap, label))
    ids = cached_dataset.reduce(ids, lambda ids, x: ids.write(ids.size(), x[2]))

    return ids.stack().numpy(), dataset

# get number of examples per label in dataset
# requires:
#   - dataset
def get_dataset_makeup(dataset)
    for i, batch in enumerate(dataset):
        for sntype in batch[1].numpy():
            relative_abundance[sntype] = 1 if sntype not in relative_abundance else relative_abundance[sntype] + 1

    return relative_abundance