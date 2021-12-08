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

from braket.circuits.free_parameter import FreeParameter


class ParameterizedOperator:
    """A parameterized operator is the abstract definition of a
    parameterizable operation for a quantum device."""

    def __init__(self):
        self._parameterized = False
        self._parameter = None

    @property
    def parameterized(self) -> bool:
        """
        Returns whether an operator is parameterized.

        Returns:
            bool: Whether an operator is parameterized.

        """
        return self._parameterized

    @property
    def parameter(self) -> FreeParameter:
        """
        Returns the free parameter associated with the object.

        Returns:
            FreeParameter: Returns the free parameter associated with the object.
        """
        return self._parameter
