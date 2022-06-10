import tensorflow as tf
import datetime
import os 


class WriteTensorboard():
    def __init__(self,
                 date_time: str,
                 tensorboard_dir: str,
                 model_prefix: str):

        self.tensorboard_dir = tensorboard_dir
        self.model_prefix = model_prefix
        self.configuration()
    
    def get_time(self):
        return datetime.datetime.now().strftime("%m%d")


    def configuration(self):
        current_time = self.get_time()

        train_log_dir = os.path.join(self.tensorboard_dir, current_time +'/', self.model_prefix + '/train')
        test_log_dir = os.path.join(self.tensorboard_dir, current_time +'/', self.model_prefix + '/valid')
        
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    
    def logging_train(self, logs: dict, epoch_step: int):
        with self.train_summary_writer.as_default():
            for _, (log_name, log_value) in enumerate(logs.items()): 
                tf.summary.scalar(log_name, log_value, step=epoch_step)


    def logging_valid(self, logs: dict, epoch_step: int):
        with self.test_summary_writer.as_default():
            for _, (log_name, log_value) in enumerate(logs.items()): 
                tf.summary.scalar(log_name, log_value, step=epoch_step)

    
    def logging_images(self, log_name: str, images, epoch_step: int):
        with self.test_summary_writer.as_default():
            num_images = images.shape[0]
            images = tf.reshape(images[0:num_images], (-1, images.shape[1], images.shape[2], 3))
            tf.summary.image(log_name, images, max_outputs=num_images, step=epoch_step)