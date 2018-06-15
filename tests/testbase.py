import unittest


class TestBase(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestBase, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertCountEqual'):
            self.assertCountEqual = self.assertItemsEqual
        