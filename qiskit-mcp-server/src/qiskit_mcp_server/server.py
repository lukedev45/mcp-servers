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

#!/usr/bin/env python3
"""
Qiskit MCP Server

A Model Context Protocol server that provides Qiskit quantum computing
capabilities, enabling AI assistants to work with quantum circuits,
transpilation, and other Qiskit features.

Dependencies:
- fastmcp
- qiskit
- python-dotenv
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_mcp_server.transpiler import (
    analyze_circuit,
    compare_optimization_levels,
    get_available_basis_gates,
    get_available_topologies,
    get_transpiler_info,
    transpile_circuit,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit")


# Tools
@mcp.tool()
async def transpile_circuit_tool(
    circuit_qasm: str,
    optimization_level: int = 2,
    basis_gates: list[str] | str | None = None,
    coupling_map: list[list[int]] | str | None = None,
    initial_layout: list[int] | None = None,
    seed_transpiler: int | None = None,
) -> dict[str, Any]:
    """Transpile a quantum circuit using Qiskit's preset pass managers.

    Takes an OpenQASM circuit and transpiles it to match target hardware
    constraints while optimizing for depth and gate count.

    Args:
        circuit_qasm: OpenQASM 2.0 or 3.0 string representation of the circuit
        optimization_level: Optimization level (0-3):
            - 0: No optimization, just maps to basis gates
            - 1: Light optimization (default mapping, simple optimizations)
            - 2: Medium optimization (noise-adaptive layout, more passes) [default]
            - 3: Heavy optimization (best results, longest compilation time)
        basis_gates: Target basis gates. Can be:
            - A list of gate names (e.g., ["cx", "id", "rz", "sx", "x"])
            - A preset name: "ibm_default", "ibm_eagle", "ibm_heron",
              "generic_clifford_t", "ion_trap", "superconducting"
            - None for no basis gate restriction
        coupling_map: Qubit connectivity. Can be:
            - A list of [control, target] pairs (e.g., [[0, 1], [1, 2]])
            - A topology name: "linear", "ring", "grid", "full"
            - None for all-to-all connectivity
        initial_layout: Optional initial qubit layout as list of physical qubit indices
        seed_transpiler: Random seed for reproducibility

    Returns:
        Dictionary with original and transpiled circuit info, and optimization metrics
    """
    return await transpile_circuit(
        circuit_qasm=circuit_qasm,
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        seed_transpiler=seed_transpiler,
    )


@mcp.tool()
async def analyze_circuit_tool(circuit_qasm: str) -> dict[str, Any]:
    """Analyze a quantum circuit without transpiling it.

    Provides detailed information about circuit structure, gate counts,
    and metrics useful for understanding circuit complexity.

    Args:
        circuit_qasm: OpenQASM 2.0 or 3.0 string representation of the circuit

    Returns:
        Dictionary with circuit analysis including gate counts, depth, and categorization
    """
    return await analyze_circuit(circuit_qasm)


@mcp.tool()
async def compare_optimization_levels_tool(circuit_qasm: str) -> dict[str, Any]:
    """Compare transpilation results across all optimization levels (0-3).

    Useful for understanding the trade-off between compilation time
    and circuit quality for a specific circuit.

    Args:
        circuit_qasm: OpenQASM 2.0 or 3.0 string representation of the circuit

    Returns:
        Dictionary comparing depth, size, and gate counts across all levels
    """
    return await compare_optimization_levels(circuit_qasm)


@mcp.tool()
async def get_available_basis_gates_tool() -> dict[str, Any]:
    """Get available preset basis gate sets.

    Returns information about predefined basis gate sets that can be
    used with the transpile_circuit tool.

    Returns:
        Dictionary with available basis gate presets and their gate lists
    """
    return await get_available_basis_gates()


@mcp.tool()
async def get_available_topologies_tool() -> dict[str, Any]:
    """Get available coupling map topologies.

    Returns information about predefined qubit connectivity topologies
    that can be used with the transpile_circuit tool.

    Returns:
        Dictionary with available topology names and descriptions
    """
    return await get_available_topologies()


@mcp.tool()
async def get_transpiler_info_tool() -> dict[str, Any]:
    """Get information about the Qiskit transpiler and available options.

    Returns comprehensive documentation about how transpilation works,
    the six transpiler stages, and usage recommendations.

    Returns:
        Dictionary with transpiler information and usage guidance
    """
    return await get_transpiler_info()


# Resources
@mcp.resource("qiskit://transpiler/info", mime_type="application/json")
async def transpiler_info_resource() -> dict[str, Any]:
    """Get Qiskit transpiler information and capabilities."""
    return await get_transpiler_info()


@mcp.resource("qiskit://transpiler/basis-gates", mime_type="application/json")
async def basis_gates_resource() -> dict[str, Any]:
    """Get available basis gate presets."""
    return await get_available_basis_gates()


@mcp.resource("qiskit://transpiler/topologies", mime_type="application/json")
async def topologies_resource() -> dict[str, Any]:
    """Get available coupling map topologies."""
    return await get_available_topologies()


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()
