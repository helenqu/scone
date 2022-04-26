import os
import numpy as np
import tensorflow as tf

# retrieves heatmap data, associated id, and label from TFRecord data files
# requires:
#   - raw_record
#   - INPUT_SHAPE
#   - CATEGORICAL
def get_images(raw_record, input_shape, has_ids=False, with_z=False):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    if has_ids: 
        image_feature_description['id'] = tf.io.FixedLenFeature([], tf.int64)
    if with_z: 
        image_feature_description['z'] = tf.io.FixedLenFeature([], tf.float32)
        image_feature_description['z_err'] = tf.io.FixedLenFeature([], tf.float32)


    example = tf.io.parse_single_example(raw_record, image_feature_description)
    image = tf.reshape(tf.io.decode_raw(example['image_raw'], tf.float64), input_shape)
    image = image / tf.reduce_max(image[:,:,0])

    # TODO: have to subtract 1 from label to get rid of KN in early classification dataset
    if with_z:
        output = [{"image": image, "z": example["z"], "z_err": example["z_err"]}, {"label": example['label']}]
    else:
        output = [{"image": image}, {"label": example['label']}]
    if has_ids:
        output.append({"id": tf.cast(example['id'], tf.int32)})
    return output

# balances classes, splits dataset into train/validation/test sets
# requires:
#   - dataset
#   - train_proportion
def stratified_split(dataset, train_proportion, types, include_test_set, class_balance):
    by_type_data_lists = {sn_type: dataset.filter(lambda image, label, *_: label["label"] == sn_type) for sn_type in types}
    by_type_data_lengths = {k: sum([1 for _ in v]) for k,v in by_type_data_lists.items()}
    print(f"number of samples per label: {by_type_data_lengths}")

    if class_balance:
        min_amount = min(by_type_data_lengths.values())
        print(f"min number of samples: {min_amount}")
        num_in_train = int(min_amount * train_proportion)
        print(f"expected train set size: {num_in_train * len(by_type_data_lengths.keys())}")
        num_in_val = int(min_amount * val_proportion)
        print(f"expected val set size: {num_in_val * len(by_type_data_lengths.keys())}")
    else:
        train_nums = {}
        val_nums = {}
        
    val_proportion = 0.5*(1-train_proportion) if include_test_set else 1-train_proportion
    
    train_set = None
    val_set = None
    test_set = None
    #TODO: make this no test set flow less ugly
      
    for sn_type, data in by_type_data_lists.items():
        # take from each with correct proportion to make stratified split train/val/test

        if not class_balance:
            num_in_train = round(by_type_data_lengths[sn_type] * train_proportion)
            num_in_val = round(by_type_data_lengths[sn_type] * val_proportion)

            train_nums[sn_type] = num_in_train
            val_nums[sn_type] = num_in_val
        
        data = data.shuffle(by_type_data_lengths[sn_type])
        current_train = data.take(num_in_train)
        current_test_val = data.skip(num_in_train)
        current_val = current_test_val if not include_test_set else current_test_val.take(num_in_val)
        current_test = None if not include_test_set else current_test_val.skip(num_in_val)

        if train_set != None:
            train_set = train_set.concatenate(current_train)
            val_set = val_set.concatenate(current_val)
            if include_test_set:
                test_set = test_set.concatenate(current_test)
        else:
            train_set = current_train 
            val_set = current_val
            if include_test_set:
                test_set = current_test

    if class_balance:
        full_dataset_size = min_amount * len(by_type_data_lists.keys()) #full dataset size = heatmaps per type * num types
    else:
        full_dataset_size = np.sum(train_nums.values()) + np.sum(val_nums.values())
        print(f"train_nums: {train_nums}")
        print(f"val_nums: {val_nums}")
        
    train_set = train_set.shuffle(full_dataset_size)
    val_set = val_set.shuffle(int(full_dataset_size*val_proportion))

    if include_test_set:
        test_set = test_set.shuffle(int(full_dataset_size*val_proportion))
    return train_set, val_set, test_set, by_type_data_lengths

# extract ids from all datasets post-caching
# requires:
#   - train_set, val_set, test_set
#   - extract_ids_from_dataset helper function
def extract_ids_and_batch(train_set, val_set, test_set, BATCH_SIZE):
    train_ids, train_set = extract_ids_from_dataset(train_set)
    val_ids, val_set = extract_ids_from_dataset(val_set)
    print("makeup of training set: {}".format(get_dataset_makeup(train_set)))
    print("makeup of validation set: {}".format(get_dataset_makeup(val_set)))
    train_set = train_set.batch(BATCH_SIZE)
    val_set = val_set.batch(BATCH_SIZE)
    if test_set:
        test_ids, test_set = extract_ids_from_dataset(test_set)
        print("makeup of test set: {}".format(get_dataset_makeup(test_set)))
        test_set = test_set.batch(BATCH_SIZE)
    else:
        test_ids = None
    
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
def get_dataset_makeup(dataset):
    relative_abundance = {}
    for i, elem in enumerate(dataset):
        sntype = elem[1].numpy()
        relative_abundance[sntype] = 1 if sntype not in relative_abundance else relative_abundance[sntype] + 1

    print(relative_abundance)
    return relative_abundance
