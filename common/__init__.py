"""Shared code used by both the Flower server and the bank clients.

Importing this package never grants access to another node's data — clients
only ever read their own mounted ``/data`` volume.
"""
