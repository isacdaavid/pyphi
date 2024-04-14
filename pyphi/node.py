#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# node.py

"""
Represents a node in a network. Each node has a unique index, its position in
the network's list of nodes.
"""

import functools

import numpy as np

from . import utils
from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .labels import NodeLabels
from .tpm import ExplicitTPM


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Args:
        forward_tpm (ExplicitTPM): The forward TPM of the subsystem.
        backward_tpm (ExplicitTPM): The backward TPM of the subsystem.
        cm (np.ndarray): The CM of the subsystem.
        index (int): The node's index in the network.
        state (int): The state of this node.
        node_labels (|NodeLabels|): Labels for these nodes.

    Attributes:
        forward_tpm (ExplicitTPM),
        backward_tpm (ExplicitTPM): The node TPM is an array with shape ``(2,)*(n + 1)``,
            where ``n`` is the size of the |Network|. The first ``n``
            dimensions correspond to each node in the system. Dimensions
            corresponding to nodes that provide input to this node are of size
            2, while those that do not correspond to inputs are of size 1, so
            that the TPM has |2^m x 2| elements where |m| is the number of
            inputs. The last dimension corresponds to the state of the node in
            the next timestep, so that ``node.tpm[..., 0]`` gives probabilities
            that the node will be 'OFF' and ``node.tpm[..., 1]`` gives
            probabilities that the node will be 'ON'.
    """

    def __init__(self, forward_tpm, backward_tpm, cm, index, state, node_labels):
        # This node's index in the list of nodes.
        self.index = index

        # State of this node.
        self.state = state

        # Node labels used in the system
        self.node_labels = node_labels

        # Get indices of the inputs.
        self._inputs = frozenset(get_inputs_from_cm(self.index, cm))
        self._outputs = frozenset(get_outputs_from_cm(self.index, cm))

        # Generate the node's TPMs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We begin by getting the part of the subsystem's TPM that gives just
        # the state of this node. This part is still indexed by network state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the network nodes.
        forward_tpm_on = forward_tpm[..., self.index]
        backward_tpm_on = backward_tpm[..., self.index]

        # TODO extend to nonbinary nodes
        # Marginalize out non-input nodes that are in the subsystem, since the
        # external nodes have already been dealt with as boundary conditions in
        # the subsystem's TPM.

        # TODO use names rather than indices
        forward_non_inputs = set(forward_tpm.tpm_indices()) - self._inputs
        forward_tpm_on = forward_tpm_on.marginalize_out(forward_non_inputs).tpm
        backward_non_inputs = set(backward_tpm.tpm_indices()) - self._inputs
        backward_tpm_on = backward_tpm_on.marginalize_out(backward_non_inputs).tpm

        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        forward_tpm_off = 1 - forward_tpm_on
        backward_tpm_off = 1 - backward_tpm_on

        # Combine the on- and off-TPM so that the first dimension is indexed by
        # the state of the node's inputs at t, and the last dimension is
        # indexed by the node's state at t+1. This representation makes it easy
        # to condition on the node state.
        self.forward_tpm = ExplicitTPM(
            np.stack([forward_tpm_off, forward_tpm_on], axis=-1),
        )
        self.backward_tpm = ExplicitTPM(
            np.stack([backward_tpm_off, backward_tpm_on], axis=-1),
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Only compute the hash once.
        self._hash = hash(
            (index, hash(self.forward_tpm), hash(self.backward_tpm), self.state,
             self._inputs, self._outputs)
        )

    @property
    def forward_tpm_off(self):
        """The forward TPM of this node containing only the 'OFF' probabilities."""
        return self.forward_tpm[..., 0]

    @property
    def backward_tpm_off(self):
        """The backward TPM of this node containing only the 'OFF' probabilities."""
        return self.backward_tpm[..., 0]

    @property
    def forward_tpm_on(self):
        """The forward TPM of this node containing only the 'ON' probabilities."""
        return self.forward_tpm[..., 1]

    @property
    def backward_tpm_on(self):
        """The backward TPM of this node containing only the 'ON' probabilities."""
        return self.backward_tpm[..., 1]

    @property
    def inputs(self):
        """The set of nodes with connections to this node."""
        return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
        return self._outputs

    @property
    def label(self):
        """The textual label for this node."""
        return self.node_labels[self.index]

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and self.forward_tpm.array_equal(other.forward_tpm)
            and self.backward_tpm.array_equal(other.backward_tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


def generate_nodes(forward_tpm, backward_tpm, cm, network_state, indices, node_labels=None):
    """Generate |Node| objects for a subsystem.

    Args:
        forward_tpm (ExplicitTPM): The system's forward_TPM
        backward_tpm (ExplicitTPM): The system's backward_TPM
        cm (np.ndarray): The corresponding CM.
        network_state (tuple): The state of the network.
        indices (tuple[int]): Indices to generate nodes for.

    Keyword Args:
        node_labels (|NodeLabels|): Textual labels for each node.

    Returns:
        tuple[Node]: The nodes of the system.
    """
    if node_labels is None:
        node_labels = NodeLabels(None, indices)

    node_state = utils.state_of(indices, network_state)

    return tuple(
        Node(forward_tpm, backward_tpm, cm, index, state, node_labels)
        for index, state in zip(indices, node_state)
    )


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    Args:
        tpm (ExplicitTPM): The node TPM to expand.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) contains only the probability
    of *this* node being on, rather than the probabilities for each node.
    """
    uc = ExplicitTPM(np.ones([2 for node in tpm.shape]))
    return uc * tpm
