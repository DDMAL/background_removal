import os

from rodan.jobs.base import RodanTask
from celery.utils.log import get_task_logger

class BgRemoval(RodanTask):

    name = 'Background Removal'
    author = 'Wanyi Lin and Khoi Nguyen'
    description = "Use Sauvola threshold to remove background"
    logger = get_task_logger(__name__)

    enabled = True
    category = 'Background removal - remove image background'
    interactive = False

    input_port_types = [{
        'name': 'Image',
        'resource_types': lambda mime: mime.startswith('image/'),
        'minimum': 1,
        'maximum': 1
    }]
    output_port_types = [{
        'name': 'RGB PNG image',
        'resource_types': ['image/rgba+png'],
        'minimum': 1,
        'maximum': 1
    }]

    settings = {
        'title': 'Remove Background',
        'type': 'object',
        'job_queue': 'Python3',

        'properties': {
            'Background Removal Method': {
                'enum': ['Sauvola Algorithm', 'SAE Binarization Model'],
                'default': 'Sauvola Algorithm',
                'description': 'Choose one method for background removal'
            },
            'window_size': {
                'type': 'integer',
                'minimum': 1,
                'default': 15
            },
            'k': {
                'type': 'number',
                'minimum': 0.0,
                'default': 0.2
            },
            'contrast': {
                'type': 'number',
                'default': 127.0
            },
            'brightness': {
                'type': 'number',
                'default': 0.0
            }
        }
    }

    def run_my_task(self, inputs, settings, outputs):
        from . import background_removal_engine as Engine
        from . import LoaderWriter
        from .binarize.binarize import run_binarize

        load_image_path = inputs['Image'][0]['resource_path']
        save_image_path = "{}.png".format(outputs['RGB PNG image'][0]['resource_path'])
        if settings['Background Removal Method'] == 0:
            image_bgr = LoaderWriter.load_image(load_image_path)
            # Remove background here.
            image_processed = Engine.remove_background(image_bgr, settings["window_size"], settings["k"], settings["contrast"], settings["brightness"])
            LoaderWriter.write_image(save_image_path, image_processed)
        else:
            image_processed = run_binarize(load_image_path, save_image_path)

        os.rename(save_image_path,outputs['RGB PNG image'][0]['resource_path'])
        return True

    def my_error_information(self, exc, traceback):
        return
