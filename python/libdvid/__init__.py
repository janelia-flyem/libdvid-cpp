from __future__ import absolute_import
from ._dvid_python import *

# There seems to be no easy way to attach
#  these member on the C++ side, and it's easy to just do it here.
@property
def DVIDException_status(self):
    return self.args[0]
DVIDException.status = DVIDException_status

@property
def DVIDException_message(self):
    return self.args[1]
DVIDException.message = DVIDException_message
