import logging
import unittest


class TestBase(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(*args, **kwargs)
        self.log_level = logging.INFO
        logging.getLogger().setLevel(self.log_level)

        if not hasattr(self, 'assertCountEqual'):
            self.assertCountEqual = self.assertItemsEqual
        