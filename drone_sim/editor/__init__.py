"""Editing stack files programmatically, for the web app's Stacks tab.

- model.py: StackDocument, a stack file as a list of node specs whose
  properties come from each node class's constructor; saving rewrites only
  the file's stack() function.
- api.py: StackEditor, JSON views (properties, laid-out graph, warnings)
  that web/src/python/bridge.py hands to the browser.
"""

from .api import StackEditor
from .model import EditError, StackDocument

__all__ = ["EditError", "StackDocument", "StackEditor"]
