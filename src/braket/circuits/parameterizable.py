# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import annotations

from typing import List, Union

from braket.circuits.free_parameter import FreeParameter


class Parameterizable:
    """A parameterized object is the definition of an object that can take in FreeParameters"""

    def __init__(self):
        self._parameters = list()

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
                associated with the object.
        """
        return self._parameters
