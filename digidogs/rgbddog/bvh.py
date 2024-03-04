'''
Based on https://github.com/behnam/cgkit/blob/master/cgkit/bvh.py

The following is the license included in the original script:

# ***** BEGIN LICENSE BLOCK *****
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# The Original Code is the Python Computer Graphics Kit.
#
# The Initial Developer of the Original Code is Matthias Baas.
# Portions created by the Initial Developer are Copyright (C) 2004
# the Initial Developer. All Rights Reserved.
#
# Contributor(s):
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# ***** END LICENSE BLOCK *****
# $Id: bvh.py,v 1.1 2005/02/06 22:26:02 mbaas Exp $

'''

import string
import sys

import json
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as Rsci    
from scipy.io import loadmat


# Node
class Node:
    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0,0,0)
        self.children = []
        self.childrenIdx = []
        self._is_root = root
        self.order = ""
        self.parentIdx = -1
        self.idx = -1
        self.position = []
        self.rotation_local_euler = []
        self.rotation_world = []
        self.rotation_local = []

    def isRoot(self):
        return self._is_root

    def isEndSite(self):
        return len(self.children)==0
    

# BVHReader
class BVHReader:
    """Read BioVision Hierarchical (BVH) files.
"""

    def __init__(self, filename):

        self.filename = filename
        # A list of unprocessed tokens (strings)
        self.tokenlist = []
        # The current line number
        self.linenr = 0

        # Root node
        self._root = None
        self._nodestack = []

        # Total number of channels
        self._numchannels = 0
        
        self.currIdx = 0

    def onHierarchy(self, root):
        pass

    def onMotion(self, frames, dt):
        pass

    def onFrame(self, values):
            pass

    # read
    def read(self, gta):
            # self.fhandle = file(self.filename)
            self.fhandle = open(self.filename, "r")
            self.readHierarchy()
            self.onHierarchy(self._root)
            self.readMotion(gta)

    # readMotion
    def readMotion(self,gta):
            """
            Read the motion samples.
            """
            # No more tokens (i.e. end of file)? Then just return
            try:
                tok = self.token()
            except StopIteration:
                return
            if tok!="MOTION":
                raise (SyntaxError, "Syntax error in line %d: 'MOTION' expected, got '%s' instead"%(self.linenr, tok))
            # Read the number of frames
            tok = self.token()
            if tok!="Frames:":
                raise (SyntaxError, "Syntax error in line %d: 'Frames:' expected, got '%s' instead"%(self.linenr, tok))
            if gta:
                frames = self.intToken()-1 
            else:
                frames = self.intToken()
            # Read the frame time
            tok = self.token()
            if tok!="Frame":
                raise (SyntaxError, "Syntax error in line %d: 'Frame Time:' expected, got '%s' instead"%(self.linenr, tok))
            tok = self.token()
            if tok!="Time:":
                raise (SyntaxError, "Syntax error in line %d: 'Frame Time:' expected, got 'Frame %s' instead"%(self.linenr, tok))
            dt = self.floatToken()
            self.onMotion(frames, dt)
            # Read the channel values
            for i in range(frames):
                s = self.readLine()
                a = s.split()
                if len(a)!=self._numchannels:
                    raise (SyntaxError, "Syntax error in line %d: %d float values expected, got %d instead"%(self.linenr, self._numchannels, len(a)))
                values = map(lambda x: float(x), a)
                self.onFrame(values)
                
    # readHierarchy
    def readHierarchy(self):
            """Read the skeleton hierarchy.
            """
            tok = self.token()
            if tok!="HIERARCHY":
                raise (SyntaxError, "Syntax error in line %d: 'HIERARCHY' expected, got '%s' instead"%(self.linenr, tok))
            
            tok = self.token()
            if tok!="ROOT":
                raise (SyntaxError, "Syntax error in line %d: 'ROOT' expected, got '%s' instead"%(self.linenr, tok))
            
            self._root = Node(root=True)
            self._nodestack.append(self._root)
            self.readNode(-1)

    # readNode
    def readNode(self, parentIdx=-1):
        """Read the data for a node.
"""

        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self._nodestack[-1].name = name
            
        id = self.currIdx
        self._nodestack[-1].idx = id
        self._nodestack[-1].parentIdx = parentIdx
        self.currIdx = self.currIdx+1
        
        tok = self.token()
        if tok!="{":
            raise (SyntaxError, "Syntax error in line %d: '{' expected, got '%s' instead"%(self.linenr, tok))

        while 1:
            tok = self.token()
            if tok=="OFFSET":
                scale = 1
                x = self.floatToken() * scale
                y = self.floatToken() * scale
                z = self.floatToken() * scale
                self._nodestack[-1].offset = (x,y,z)
            elif tok=="CHANNELS":
                n = self.intToken()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                  "Xrotation", "Yrotation", "Zrotation"]:
                        raise (SyntaxError, "Syntax error in line %d: Invalid channel name: '%s'"%(self.linenr, tok))
                    channels.append(tok)
                self._numchannels += len(channels)
                self._nodestack[-1].channels = channels
            elif tok=="JOINT":
                node = Node()
                self._nodestack[-1].children.append(node)
                self._nodestack.append(node)
                self.readNode(id)
            elif tok=="End":
                node = Node()
                self._nodestack[-1].children.append(node)
                self._nodestack.append(node)
                self.readNode(id)
            elif tok=="}":
                if self._nodestack[-1].isEndSite():
                    self._nodestack[-1].name = "End Site"
                self._nodestack.pop()
                break
            else:
                raise (SyntaxError, "Syntax error in line %d: Unknown keyword '%s'"%(self.linenr, tok))
        

    # intToken
    def intToken(self):
        """Return the next token which must be an int.
"""

        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise (SyntaxError, "Syntax error in line %d: Integer expected, got '%s' instead"%(self.linenr, tok))

    # floatToken
    def floatToken(self):
        """Return the next token which must be a float.
"""

        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise (SyntaxError, "Syntax error in line %d: Float expected, got '%s' instead"%(self.linenr, tok))

    # token
    def token(self):
        """Return the next token."""

        # Are there still some tokens left? then just return the next one
        if self.tokenlist!=[]:
            tok = self.tokenlist[0]
            self.tokenlist = self.tokenlist[1:]
            return tok

        # Read a new line
        s = self.readLine()
        self.createTokens(s)
        return self.token()

    # readLine
    def readLine(self):
        """Return the next line.

        Empty lines are skipped. If the end of the file has been
        reached, a StopIteration exception is thrown. The return
        value is the next line containing data (this will never be an
        empty string).
        """
        # Discard any remaining tokens
        self.tokenlist = []
      
        # Read the next line
        while 1:
            s = self.fhandle.readline()
            self.linenr += 1
            if s=="":
                pass
                #raise( StopIteration)
            return s

    # createTokens
    def createTokens(self, s):
        """Populate the token list from the content of s.
"""
        s = s.strip()
        a = s.split()
        self.tokenlist = a
