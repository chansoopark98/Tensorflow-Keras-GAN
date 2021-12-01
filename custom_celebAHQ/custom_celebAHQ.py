"""custom_celebAHQ dataset."""

import tensorflow_datasets as tfds
import glob
import os

# TODO(custom_celebAHQ): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(custom_celebAHQ): BibTeX citation
_CITATION = """
"""


class CustomCelebahq(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for custom_celebAHQ dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/kp/tensorflow_datasets/downloads/manual'

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(custom_celebAHQ): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.Image(shape=(None, None, 3)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(custom_celebAHQ): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'data.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(custom_celebAHQ): Returns the Dict[split names, Iterator[Key, Example]]
    # TODO(custom_celebA): Returns the Dict[split names, Iterator[Key, Example]]
    return {'train': self._generate_examples(img_path=extracted_path/'train')}

  def _generate_examples(self, img_path):
    """Yields examples."""
    # TODO(custom_celebAHQ): Yields (key, example) tuples from the dataset
    img = os.path.join(img_path, '*.jpg')
    img_files = glob.glob(img)

    for i in range(len(img_files)):
        yield i, {
            'image': img_files[i],
            'label': img_files[i]
        }