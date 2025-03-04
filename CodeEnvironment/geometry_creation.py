import System
import Rhino
import Grasshopper

import scriptcontext
from rhinoscript import utility as rhutil
from rhinoscript import object as rhobject
import rhinoscriptsyntax as rs

def importPoint(point3d_object):
    point_list = (point3d_object.X, point3d_object.Y, point3d_object.Z)
    return point_list
    

def getLength(point3d_object):
    return (point3d_object.X*point3d_object.X + point3d_object.Y*point3d_object.Y + point3d_object.Z*point3d_object.Z)**0.5

def CreateBox(Origin, x_size, y_size, z_size):
    # Calculate the differences in each dimension
    dx = x_size
    dy = y_size
    dz = z_size
    # Define the 8 corners of the box based on Origin and Endpoint
    corner_points = [
    (Origin),    
    (Origin[0] + dx, Origin[1], Origin[2]),    # Bottom front left
    (Origin[0] + dx, Origin[1] + dy, Origin[2]),   # Bottom front right
    (Origin[0], Origin[1] + dy, Origin[2]),  # Bottom back right
    (Origin[0], Origin[1], Origin[2] + dz),   # Bottom back left
    (Origin[0] + dx, Origin[1], Origin[2] + dz),   # Top front left
    (Origin[0] + dx, Origin[1] + dy, Origin[2] + dz),  # Top front right
    (Origin[0], Origin[1] + dy, Origin[2] + dz), # Top back right
    (0, 20, 30)   # Top back left
    ]
    # The points need to be in counter-clockwise order starting with the bottom rectangle of the box 
    # Placeholder for box creation command, as we're simulating
    # Normally, here you would use a command like rs.AddBox(points)
    return rs.AddBox(corner_points)

def CreateBox2pt(Origin, Endpoint):
    # Calculate the differences in each dimension
    dx = Endpoint[0] - Origin[0]
    dy = Endpoint[1] - Origin[1]
    dz = Endpoint[2] - Origin[2]
    # Define the 8 corners of the box based on Origin and Endpoint
    corner_points = [
    (Origin),    
    (Origin[0] + dx, Origin[1], Origin[2]),    # Bottom front left
    (Origin[0] + dx, Origin[1] + dy, Origin[2]),   # Bottom front right
    (Origin[0], Origin[1] + dy, Origin[2]),  # Bottom back right
    (Origin[0], Origin[1], Origin[2] + dz),   # Bottom back left
    (Origin[0] + dx, Origin[1], Origin[2] + dz),   # Top front left
    (Origin[0] + dx, Origin[1] + dy, Origin[2] + dz),  # Top front right
    (Origin[0], Origin[1] + dy, Origin[2] + dz), # Top back right
    (0, 20, 30)   # Top back left
    ]
    # The points need to be in counter-clockwise order starting with the bottom rectangle of the box 
    # Placeholder for box creation command, as we're simulating
    # Normally, here you would use a command like rs.AddBox(points)
    return rs.AddBox(corner_points)

def CreateCylinder(Start, End, radius, cap = True):
    End = rhutil.coerce3dpoint(End, True) #ist damit Rhino richtige punkte macht
    Start = rhutil.coerce3dpoint(Start, True)
    height = getLength(End-Start)
    normal = End - Start
    plane = Rhino.Geometry.Plane(Start, normal)
    circle = Rhino.Geometry.Circle(plane, radius)
    cylinder = Rhino.Geometry.Cylinder(circle, height)
    brep = cylinder.ToBrep(cap, cap)
    #Create the Item and add an ID to it
    id = scriptcontext.doc.Objects.AddBrep(brep)
    if id==System.Guid.Empty: return scriptcontext.errorhandler()
    scriptcontext.doc.Views.Redraw()
    return id