# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for circuit serialization utilities."""

import pytest
from qiskit import QuantumCircuit
from qiskit_mcp_server.circuit_serialization import (
    dump_circuit,
    dump_qasm_circuit,
    dump_qpy_circuit,
    load_circuit,
    load_qasm_circuit,
    load_qpy_circuit,
)


@pytest.fixture
def simple_circuit():
    """Create a simple quantum circuit for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def valid_qasm3():
    """Valid QASM 3.0 string."""
    return """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""


@pytest.fixture
def invalid_qasm3():
    """Invalid QASM 3.0 string."""
    return "INVALID QASM STRING { this is not valid }"


class TestLoadQasmCircuit:
    """Tests for load_qasm_circuit function."""

    def test_load_valid_qasm(self, valid_qasm3):
        """Test loading a valid QASM 3.0 string."""
        result = load_qasm_circuit(valid_qasm3)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit"].num_qubits == 2

    def test_load_invalid_qasm(self, invalid_qasm3):
        """Test loading an invalid QASM 3.0 string."""
        result = load_qasm_circuit(invalid_qasm3)

        assert result["status"] == "error"
        assert "message" in result
        assert "QASM 3.0 string not valid" in result["message"]


class TestLoadQpyCircuit:
    """Tests for load_qpy_circuit function."""

    def test_load_valid_qpy(self, simple_circuit):
        """Test loading a valid QPY string."""
        # First dump the circuit to QPY
        qpy_str = dump_qpy_circuit(simple_circuit)

        # Then load it back
        result = load_qpy_circuit(qpy_str)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)
        assert result["circuit"].num_qubits == simple_circuit.num_qubits

    def test_load_invalid_qpy(self):
        """Test loading an invalid QPY string."""
        result = load_qpy_circuit("not-valid-base64!")

        assert result["status"] == "error"
        assert "message" in result
        assert "Invalid QPY data" in result["message"]

    def test_load_invalid_base64_qpy(self):
        """Test loading valid base64 but invalid QPY data."""
        import base64

        invalid_data = base64.b64encode(b"not qpy data").decode("utf-8")
        result = load_qpy_circuit(invalid_data)

        assert result["status"] == "error"
        assert "Invalid QPY data" in result["message"]


class TestDumpQasmCircuit:
    """Tests for dump_qasm_circuit function."""

    def test_dump_circuit_to_qasm(self, simple_circuit):
        """Test dumping a circuit to QASM 3.0."""
        qasm_str = dump_qasm_circuit(simple_circuit)

        assert isinstance(qasm_str, str)
        assert "OPENQASM" in qasm_str
        # Should contain gate operations
        assert "h " in qasm_str.lower() or "h(" in qasm_str.lower()


class TestDumpQpyCircuit:
    """Tests for dump_qpy_circuit function."""

    def test_dump_circuit_to_qpy(self, simple_circuit):
        """Test dumping a circuit to QPY."""
        qpy_str = dump_qpy_circuit(simple_circuit)

        assert isinstance(qpy_str, str)
        # QPY base64 strings are typically alphanumeric with + / =
        import base64

        # Should be valid base64
        decoded = base64.b64decode(qpy_str)
        assert len(decoded) > 0


class TestLoadCircuit:
    """Tests for unified load_circuit function."""

    def test_load_circuit_qasm3_default(self, valid_qasm3):
        """Test load_circuit defaults to QASM3 format."""
        result = load_circuit(valid_qasm3)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_qasm3_explicit(self, valid_qasm3):
        """Test load_circuit with explicit QASM3 format."""
        result = load_circuit(valid_qasm3, circuit_format="qasm3")

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_circuit_qpy(self, simple_circuit):
        """Test load_circuit with QPY format."""
        qpy_str = dump_qpy_circuit(simple_circuit)
        result = load_circuit(qpy_str, circuit_format="qpy")

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)


class TestDumpCircuit:
    """Tests for unified dump_circuit function."""

    def test_dump_circuit_qasm3_default(self, simple_circuit):
        """Test dump_circuit defaults to QASM3 format."""
        result = dump_circuit(simple_circuit)

        assert isinstance(result, str)
        assert "OPENQASM" in result

    def test_dump_circuit_qasm3_explicit(self, simple_circuit):
        """Test dump_circuit with explicit QASM3 format."""
        result = dump_circuit(simple_circuit, circuit_format="qasm3")

        assert isinstance(result, str)
        assert "OPENQASM" in result

    def test_dump_circuit_qpy(self, simple_circuit):
        """Test dump_circuit with QPY format."""
        result = dump_circuit(simple_circuit, circuit_format="qpy")

        assert isinstance(result, str)
        # QPY is base64, won't contain OPENQASM
        assert "OPENQASM" not in result


class TestRoundTrip:
    """Tests for round-trip serialization/deserialization."""

    def test_qasm3_round_trip(self, simple_circuit):
        """Test QASM3 round-trip preserves circuit structure."""
        # Dump to QASM3
        qasm_str = dump_circuit(simple_circuit, circuit_format="qasm3")

        # Load back
        result = load_circuit(qasm_str, circuit_format="qasm3")

        assert result["status"] == "success"
        loaded_circuit = result["circuit"]

        # Check structure is preserved
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert loaded_circuit.depth() == simple_circuit.depth()

    def test_qpy_round_trip(self, simple_circuit):
        """Test QPY round-trip preserves circuit exactly."""
        # Dump to QPY
        qpy_str = dump_circuit(simple_circuit, circuit_format="qpy")

        # Load back
        result = load_circuit(qpy_str, circuit_format="qpy")

        assert result["status"] == "success"
        loaded_circuit = result["circuit"]

        # Check structure is preserved
        assert loaded_circuit.num_qubits == simple_circuit.num_qubits
        assert loaded_circuit.depth() == simple_circuit.depth()
        # QPY should preserve gate count exactly
        assert len(loaded_circuit.data) == len(simple_circuit.data)

    def test_qpy_preserves_parameters(self):
        """Test QPY preserves parameterized gates with exact values."""
        import math

        qc = QuantumCircuit(1)
        qc.rx(math.pi / 7, 0)  # Use an angle that might lose precision in QASM3

        # Round-trip through QPY
        qpy_str = dump_circuit(qc, circuit_format="qpy")
        result = load_circuit(qpy_str, circuit_format="qpy")

        loaded_circuit = result["circuit"]
        original_angle = qc.data[0].operation.params[0]
        loaded_angle = loaded_circuit.data[0].operation.params[0]

        # QPY should preserve exact floating point value
        assert original_angle == loaded_angle

    def test_qpy_preserves_circuit_name(self):
        """Test QPY preserves circuit metadata like name."""
        qc = QuantumCircuit(2, name="my_test_circuit")
        qc.h(0)

        # Round-trip through QPY
        qpy_str = dump_circuit(qc, circuit_format="qpy")
        result = load_circuit(qpy_str, circuit_format="qpy")

        loaded_circuit = result["circuit"]
        assert loaded_circuit.name == "my_test_circuit"
