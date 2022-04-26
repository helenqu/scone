import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers
import h5py
import time
import json
import argparse

from data_utils import *

class SconeClassifier():
    # define my own reshape layer
    class Reshape(layers.Layer):
        def call(self, inputs):
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self): # for model saving/loading
            return {}

    def __init__(self, config):
        self.heatmaps_path = config['heatmaps_path']
        self.mode = config["mode"]
        
        self.strategy = tf.distribute.MirroredStrategy()
        self.batch_size_per_replica = config.get('batch_size', 32)
        self.batch_size = self.batch_size_per_replica * self.strategy.num_replicas_in_sync
        print(f"batch size in config: {self.batch_size_per_replica}, num replicas: {self.strategy.num_replicas_in_sync}, true batch size: {self.batch_size}")

        self.num_epochs = config['num_epochs']
        self.input_shape = (config['num_wavelength_bins'], config['num_mjd_bins'], 2)
        self.categorical = config['categorical']
        self.types = config.get('types', None)
        if self.categorical and self.types is None:
            raise KeyError('cannot perform categorical classification without knowing the number of source types! please specify the `types` key in your config file to reflect this information')
            # TODO: should i write num types info into a file after create heatmaps? maybe ids file will be large
            # ids_file = h5py.File(config['ids_path'], "r")
            # types = [x.decode('utf-8').split("_")[0] for x in ids_file["names"]]
            # ids_file.close()
            # self.num_types = len(np.unique(types))
        self.num_types = len(self.types) if self.categorical else 2
        self.train_proportion = config.get('train_proportion', 0.8)
        self.has_ids = config.get('has_ids', False)
        self.with_z = config.get('with_z', False)
        self.predict = True if self.mode == "predict" else False
        self.external_trained_model = config.get("trained_model")
        self.abundances = None
        self.train_set = self.val_set = self.test_set = None
        self.class_balanced = config.get('class_balanced', True)
        
    @staticmethod
    def _print_report_and_save_history(history, start_time, path):
        print("######## CLASSIFICATION REPORT ########")
        if "accuracy" in history:
            print("classification finished in {:.2f}min".format((time.time() - start_time) / 60))
            print("last training accuracy value: {}".format(history["accuracy"][-1]))
            print("last validation accuracy value: {}".format(history["val_accuracy"][-1]))
        if "test_accuracy" in history:
            print("test accuracy value: {}".format(history["test_accuracy"]))

        with open(os.path.join(path, "history.json"), 'w') as outfile:
            json.dump(history, outfile)

    def run(self):
        start = time.time()

        self.train_set, self.val_set, self.test_set = self._split_and_retrieve_data()

        if self.external_trained_model:
            print(f"loading trained model found at {self.external_trained_model}")
            self.trained_model = models.load_model(self.external_trained_model, custom_objects={"Reshape": self.Reshape})

        if self.mode == "train":
            self.trained_model, history = self.train()
            history = history.history

        if self.predict:
            print("running prediction on test set")
            preds_dict = self.predict(self.test_set)
            pd.DataFrame(preds_dict).to_csv(os.path.join(self.heatmaps_path, "preds.csv"), index=False)

            test_acc = self.test()
            history["test_accuracy"] = test_acc

        _print_report_and_save_history(history, start, self.heatmaps_path)

    # train the model, returns trained model & training log
    # requires:
    #   - model
    #   - train_set, val_set
    #   - NUM_EPOCHS
    #   - batch_size
    def train(self, train_set=None, val_set=None):
        with self.strategy.scope():
            model = self._define_and_compile_model() if not self.external_trained_model else self.trained_model
            print(model.summary())
            train_set = train_set if train_set is not None else self.train_set
            val_set = val_set if val_set is not None else self.val_set

        if not self.class_balanced:
            print("not class balanced, applying class weights")
            class_weights = {k: (self.batch_size / (self.num_types * v)) for k,v in self.abundances.items()}

        train_set = train_set.map(lambda image, label, *args: (image, label)).shuffle(100_000).cache().batch(self.batch_size)
        val_set = val_set.map(lambda image, label, *args: (image, label)).shuffle(10_000).cache().batch(self.batch_size)
        print("starting to train")
        history = model.fit(
            train_set,
            epochs=self.num_epochs,
            validation_data=val_set,
            verbose=1,
            class_weight=class_weights if not self.class_balanced else None)

        model.save(f"{self.heatmaps_path}/trained_model")
        return model, history

    def predict(self, dataset, dataset_ids=None):
        if self.external_trained_model and not self.trained_model:
            self.trained_model = models.load_model(self.external_trained_model, custom_objects={"Reshape": self.Reshape})

        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before predict!')
        
        dataset = dataset.cache() # otherwise the rest of the dataset operations won't return entries in the same order
        dataset_no_ids = dataset.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        predictions = self.trained_model.predict(dataset_no_ids, verbose=0)
        if self.categorical:
            predictions = np.argmax(predictions, axis=1) #TODO: is this the best way to return categorical results? doesnt preserve confidence info
        predictions = predictions.flatten()

        true_labels = dataset.map(lambda _, label, *args: label["label"])
        df_dict = {'pred_labels': predictions, 'true_labels': list(true_labels.as_numpy_iterator())}
        if self.has_ids:
            ids = dataset.map(lambda _, label, id_: id_["id"])
            df_dict['snid'] = list(ids.as_numpy_iterator())
        # if dataset_ids is not None:
        #     df_dict['snid'] = dataset_ids
        return df_dict

    def test(self, test_set=None):
        if not self.predict:
            raise RuntimeError('no testing in train mode')
        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before test!')
        test_set = test_set or self.test_set
        test_set = test_set.map(lambda image, label, *_: (image, label)).batch(self.batch_size)
        results = self.trained_model.evaluate(test_set)
        return results[1]

    # defines and compiles, then returns model
    # requires:
    #   - INPUT_SHAPE
    #   - CATEGORICAL
    #   - NUM_TYPES
    def _define_and_compile_model(self, metrics=['accuracy']):
        y, x, _ = self.input_shape

        image_input = tf.keras.Input(shape=self.input_shape, name="image")
        # z_input, z_err_input will only be used when doing classification with redshift
        z_input = tf.keras.Input(shape=(1,), name="z")
        z_err_input = tf.keras.Input(shape=(1,), name="z_err")
        inputs = [image_input] if not self.with_z else [image_input, z_input, z_err_input]
        
        x = layers.ZeroPadding2D(padding=(0,1))(image_input)
        x = layers.Conv2D(y, (y, 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)
        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(y, (y, 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((2, 2))(x)
       
        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)
        x = layers.ZeroPadding2D(padding=(0,1))(x)
        x = layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu')(x)
        x = self.Reshape()(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        if self.with_z:
            x = layers.concatenate([x, z_input, z_err_input])
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        if self.categorical:
            sn_type_pred = layers.Dense(self.num_types, activation='softmax', name="label")(x)
        else:
            sn_type_pred = layers.Dense(1, activation='sigmoid', name="label")(x)
        
        model = models.Model(inputs=inputs, outputs=[sn_type_pred])
        opt = optimizers.Adam(learning_rate=5e-5)
        loss = 'sparse_categorical_crossentropy' if self.categorical else 'binary_crossentropy'
        print(metrics)
        model.compile(optimizer=opt,
                      loss=loss,
                      metrics=metrics)

        return model

    def _load_dataset(self):
        filenames = ["{}/{}".format(self.heatmaps_path, f.name) for f in os.scandir(self.heatmaps_path) if "tfrecord" in f.name]
        np.random.shuffle(filenames)
        print(len(filenames))
        raw_dataset = tf.data.TFRecordDataset(
            filenames,
            num_parallel_reads=80)

        return raw_dataset

    def _retrieve_data(self, raw_dataset):
        dataset_size = sum([1 for _ in raw_dataset])
        dataset = raw_dataset.map(lambda x: get_images(x, self.input_shape, self.has_ids, self.with_z), num_parallel_calls=40)
        # self.types = [0,1] if not self.categorical else range(0, self.num_types)

        return dataset.apply(tf.data.experimental.ignore_errors()), dataset_size

    # simpler split and retrieve function
    # - doesn't explicitly split by type (i.e. works better for already class-balanced data)
    # - doesn't do class balancing
    def _split_and_retrieve_data(self):
        dataset, size = self._retrieve_data(self._load_dataset())
        dataset = dataset.shuffle(size)

        unique, counts = np.unique(list(dataset.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        print(f"dataset abundances: {dict(zip(unique, counts))}")
        num_per_type = min(counts)

        train_set_size_per_type = int(num_per_type*self.train_proportion)
        val_test_proportion = 1-self.train_proportion
        val_test_set_size_per_type = int(num_per_type*val_test_proportion)

        for i in range(self.num_types):
            filtered = dataset.filter(lambda image, label, *_: label["label"] == i)
            curr_train_set = filtered.take(train_set_size_per_type)
            curr_val_test_set = filtered.skip(train_set_size_per_type).take(val_test_set_size_per_type)
            curr_val_set = curr_val_test_set.take(val_test_set_size_per_type//2)
            curr_test_set = curr_val_test_set.skip(val_test_set_size_per_type//2).take(val_test_set_size_per_type//2)

            if i == 0:
                train_set = curr_train_set
                val_set = curr_val_set if self.predict else curr_val_test_set
                test_set = curr_test_set
            else:
                train_set = train_set.concatenate(curr_train_set)
                val_set = val_set.concatenate(curr_val_set) if self.predict else val_set.concatenate(curr_val_test_set)
                test_set = test_set.concatenate(curr_test_set)


        # train_set = dataset.take(train_set_size)
        unique, counts = np.unique(list(train_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        print(f"train set abundances: {dict(zip(unique, counts))}")

        unique, counts = np.unique(list(val_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        print(f"val set abundances: {dict(zip(unique, counts))}")

        unique, counts = np.unique(list(test_set.map(lambda image, label, *_: label["label"]).as_numpy_iterator()), return_counts=True)
        print(f"test set abundances: {dict(zip(unique, counts))}")

        return train_set, val_set, test_set

    # main data retrieval function
    # requires:
    #   - heatmaps_path
    #   - get_images
    #   - stratified_split
    #   - extract_ids
    def _split_and_retrieve_data_stratified(self):
        dataset, size = self._retrieve_data(self._load_dataset())

        train_set, val_set, test_set, self.abundances = stratified_split(dataset, self.train_proportion, self.types, self.predict, self.class_balanced)

        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if self.predict else None

        if self.has_ids:
            # TODO: maybe don't actually extract the ids - too time-consuming, not that useful
            return extract_ids_and_batch(train_set, val_set, test_set, self.batch_size)
        else:
            train_set = train_set.batch(self.batch_size)
            val_set = val_set.batch(self.batch_size)
            test_set = test_set.batch(self.batch_size)
            return train_set, val_set, test_set


class SconeClassifierIaModels(SconeClassifier):
    # define my own reshape layer
    class Reshape(layers.Layer):
        def call(self, inputs):
            return tf.transpose(inputs, perm=[0,3,2,1])

        def get_config(self): # for model saving/loading
            return {}

    def __init__(self, config):
        super().__init__(config)
        self.external_test_sets = config.get('external_test_sets', None)

    def run(self):
        if not self.trained_model:
            _, history = self.train()
        dataset, dataset_ids = self.test_set if self.predict else self.train_set
        preds_dict = self.predict(dataset, dataset_ids)

        if self.predict:
            test_acc = self.test()
            history.history["test_accuracy"] = test_acc

        if self.external_test_sets:
            for test_set_dir in self.external_test_sets:
                Ia_dataset = tf.data.TFRecordDataset(
                    ["{}/{}".format(test_set_dir, f.name) for f in os.scandir(test_set_dir) if "tfrecord" in f.name],
                    num_parallel_reads=80)

                Ia_dataset = Ia_dataset.take(20_000)
                raw_dataset = Ia_dataset.concatenate(self.non_Ia_dataset)

                dataset = raw_dataset.map(lambda x: get_images(x, self.input_shape, self.categorical, self.has_ids), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.apply(tf.data.experimental.ignore_errors())
                dataset = dataset.shuffle(20000).cache()
                ood_ids, dataset = extract_ids_from_dataset(dataset)
                test_set = dataset.batch(self.batch_size)

                accuracy = self.test(test_set)
                history.history[os.path.basename(test_set_dir) + "_test_accuracy"] = accuracy

        return preds_dict, history

    def _split_and_retrieve_data(self):
        Ia_dataset, nonIa_dataset = self._load_dataset_separate_Ia_non()
        raw_dataset = Ia_dataset.concatenate(non_Ia_dataset).shuffle(400_000)
        dataset = self._retrieve_data(raw_dataset)

        train_set, val_set, test_set = stratified_split(dataset, self.train_proportion, self.types, self.predict)
        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if self.predict else None

        if self.has_ids:
            return extract_ids_and_batch(train_set, val_set, test_set, self.batch_size)
        else:
            train_set = train_set.batch(self.batch_size)
            val_set = val_set.batch(self.batch_size)
            test_set = test_set.batch(self.batch_size)
            return train_set, val_set, test_set

    def _load_dataset_separate_Ia_non(self):
        nonIa_loc = [path for path in self.heatmaps_path if "non" in path][0]
        Ia_loc = [path for path in self.heatmaps_path if path != nonIa_loc][0]

        Ia_dataset = tf.data.TFRecordDataset(
	    ["{}/{}".format(Ia_loc, f.name) for f in os.scandir(Ia_loc) if "tfrecord" in f.name],
	    num_parallel_reads=80)
        non_Ia_dataset = tf.data.TFRecordDataset(
            ["{}/{}".format(nonIa_loc, f.name) for f in os.scandir(nonIa_loc) if "tfrecord" in f.name],
            num_parallel_reads=80) 

        Ia_dataset = Ia_dataset.shuffle(100_000)
        non_Ia_dataset = non_Ia_dataset.shuffle(100_000)
        return Ia_dataset, non_Ia_dataset


if __name__ == "__main__":
    def load_config(config_path):
        with open(config_path, "r") as cfgfile:
            config = yaml.load(cfgfile)
        return config

    parser = argparse.ArgumentParser(description='set up the SCONE model')
    parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/config.yml"')
    args = parser.parse_args()

    scone_config = load_config(args.config_path)
    SconeClassifier(scone_config).run()
