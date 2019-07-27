#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                              RattlePy Toolbox
 
  Author: Manuel Blanco Valentín 
           (mbvalentin@cbpf.br / manuel.blanco.valentin@gmail.com)

  Collaborators:   Clécio de Bom
                   Luciana Dias

  Sponsors and legal owners: PETROBRÁS
                             CBPF (Centro Brasileiro de Pesquisas Físicas)


  Copyright 2019  
  
  This program is property software; you CANNOT redistribute it and/or modify
  it without the explicit permission of the authors, collaborators and its 
  legal owners (PETROBRÁS and CBPF). Disobeying these guidelines will lead 
  to a violation of the private intellectual property of the owners, along 
  with the legal reprecaussion that this violation might cause.
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
ABOUT THE ADVANCED MODEL PLOT CODE:
    This module is a toolbox for creating nice graphics of keras models.
"""

"""
Basic Modules
"""
import re, time, sys, os
import numpy as np
from ast import literal_eval

"""
Keras utilities and layers
"""
from keras.utils.vis_utils import model_to_dot

"""
Fonts handling
"""
from PIL import ImageFont

""" Load references """
import json
css_file = os.sep.join(['utils','__res__','css','keras_layers_styles.json'])
fonts_dir = os.sep.join(['utils','__res__','fonts'])

css_refs = dict()
if os.path.isfile(css_file):
    with open(css_file,'r') as fjson:
        css_refs = json.load(fjson)

""" Function to convert from hex value to RGB equivalent """
def _hex2RGB(h):
    if h[0] == "#":
        h = h[1:]
    R = int("0x"+h[0:2],16)
    G = int("0x"+h[2:4],16)
    B = int("0x"+h[4:6],16)
    A = 255
    if len(h) == 8:
        A = int("0x"+h[6:8],16)
    return R,G,B,A


""" Required Compatible SVG header """
def _SVG_init(width,height,x0=0,y0=0,docname='drawing.svg'):
    
    SVG = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    SVG += '<!-- Created with RattlePy (https://github.com/manuelblancovalentin/rattlepy) -->\n'
    SVG += '<svg '\
            '\txmlns:dc="http://purl.org/dc/elements/1.1/"\n '\
            '\txmlns:cc="http://creativecommons.org/ns#"\n '\
            '\txmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n '\
            '\txmlns:svg="http://www.w3.org/2000/svg"\n '\
            '\txmlns="http://www.w3.org/2000/svg"\n '\
            '\txmlns:xlink="http://www.w3.org/1999/xlink"\n '\
            '\txmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"\n '\
            '\txmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n '\
            '\tviewBox="{} {} {} {}"\n '.format(x0,y0,x0+width,y0+height) + ''\
            '\tversion="1.1"\n '\
            '\tid="{}"\n '.format(docname.replace('_','').replace('.svg','')) + '' \
            '\tinkscape:version="0.92.3 (2405546, 2018-03-11)"\n '\
            '\tsodipodi:docname="{}">\n '.format(docname) + '' \
            '\t<defs id="defs2"> \n'\
        	'\t\t<style type="text/css">@font-face { font-family: Ubuntu Regular; '\
            'src: url("https://github.com/rvagg/nodei.co/blob/master/fonts/UbuntuMono-R.ttf");}'\
        	'\t\t</style> \n'\
            '\t</defs>\n'\
            '\t<defs id="defs3"><marker id="arrow" markerWidth="10" '\
            'markerHeight="10" refX="0" refY="3" orient="auto" '\
            'markerUnits="strokeWidth"> <path d="M0,1.5 L0,4.5 L3.5,3 z" '\
            'fill="#000000" /> </marker>\n'\
            '\t</defs>\n'

    return SVG


""" This class defines a SVG Tag for any input keras layer """
class layerTag(object):
    def __init__(self, layer = None, layer_type = None, 
                 center = (0,0), width = 0, height = 0,
                 background_color = css_refs['layers']['Common']['background_color'],
                 border_color = css_refs['layers']['Common']['border_color'],
                 font_color = css_refs['layers']['Common']['font_color'],
                 tag = None, parameters_references = {}, kind = 'rounded_box', 
                 **kwargs):
        
        super(layerTag,self).__init__()
        
        """ Init structure """
        self.wrapper = layer if hasattr(layer,'layer') else None
        self.wrapper_type = layer.__class__.__name__ if hasattr(layer,'layer') else None
        self.layer = layer.layer if hasattr(layer,'layer') else layer
        self.layer_type = layer_type
        self.center = center
        self.width = width
        self.height = height
        self.background_color = background_color
        self.border_color = border_color
        self.font_color = font_color
        self.tag = tag
        self.parameters_references = parameters_references
        self.parameters = None
        self.kind = kind
        
        """ Svg tag """
        self.svg_tag = None
        
        """ If we actually have a keras layer let's update the values """
        self._update_values()
        self._compute_shape()
        
    """ function to get all values from layer """
    def _update_values(self):
        if self.layer is not None:
            layer_type = self.layer.__class__.__name__
            if layer_type not in css_refs['layers']:
                layer_type = 'Common'
            self.layer_type = layer_type
        if self.layer_type is not None:
            css_entry = css_refs['layers'][self.layer_type]
            
            # params
            self.background_color = css_entry['background_color']
            self.border_color = css_entry['border_color']
            self.font_color = css_entry['font_color']
            self.parameters_references = css_entry['parameters']
            self.kind = css_entry['kind']
            if self.tag is None:
                self.tag = css_entry['tag']
            if self.tag is not None:
                try:
                    if hasattr(eval(self.tag),'__call__') and 'lambda' in self.tag:
                        try:
                            self.tag = eval(self.tag)(self.layer)
                        except:
                            self.tag = self.layer_type
                except:
                    True
                
                # gfet params
                self._get_parameters()
            
    """ function to get the parameters str from the layer (if layer exists) """
    def _get_parameters(self):
        parameters = {}
        if self.layer is not None:
            for att in self.parameters_references:
                if hasattr(self.layer,att):
                    parameters[att] = eval(literal_eval(self.parameters_references[att]))(self.layer)
    
    """ Compute shape """
    def _compute_shape(self):
        margin = (20, 10) 
        if self.tag is not None:
            font = ImageFont.truetype(os.path.join(fonts_dir,
                     css_refs['globals']['font_tag']['name']), 
                     int(css_refs['globals']['font_tag']['font_size'].replace('px','')))
            
            tagSize = font.getsize(self.tag) if not isinstance(self.tag,list) \
                       else np.max(np.array([font.getsize(tt) for tt in self.tag]),\
                                   axis=0)
                       
            self.width = tagSize[0] + 2*margin[0]
            self.height = tagSize[1] + 2*margin[1]
    
    """ function to build the svg tag """
    def _build_svg(self):
        if not isinstance(self.parameters, dict):
            self._get_parameters()

        """ Parse if this is a special layer or not """
        if self.kind == 'circle':
            # object
            SVG = ['<g>']
            SVG.append('\t<circle cx="{}" cy="{}" r="30" stroke="{}" '\
                       'stroke-width="5" fill="{}" fill-opacity="{:1.2f}"/>'.format(self.center[0],
                                                            self.center[1],
                                                             self.border_color[:-2],
                                                            self.background_color[:-2],
                                                            _hex2RGB(self.background_color)[3]/255.))
            
            # text
            # I know this is ugly, but there's no way to scape this:
            dy = 0
            fs = "50px"
            if self.tag == '⌒':
                dy = 15
                fs = "40px"
            SVG.append('\t<text x="{}" y="{}" text-anchor="middle" '\
                       'fill="{}" fill-opacity="{:1.2f}" font-size="{}" '\
                       'font-family="Ubuntu Light" dy=".3em">{}</text>'.format(self.center[0],
                                                            self.center[1]+dy,
                                                            self.font_color[:-2],
                                                            _hex2RGB(self.font_color)[3]/255.,
                                                            fs,
                                                            self.tag
                                                            ))
            SVG.append('</g>')
        
        elif 'box' in self.kind:
            # object
            SVG = ['<g>']
            # outter rectangle
            H,W = self.height, self.width
            b = css_refs['globals']['rounded_box_border_size']
            br = css_refs['globals']['rounded_box_border_radius']
            if self.kind != 'rounded_box':
                br = 0
                
            # outter rectangle
            SVG.append('\t<rect x="{}" y="{}" width="{}" height="{}" '\
                       'rx="{}" ry="{}" fill="{}" '\
                       'fill-opacity="{:1.2f}" />'.format(self.center[0] - W/2 - b,
                                                    self.center[1] - H/2 - b,
                                                    W + 2*b,
                                                    H + 2*b,
                                                    1.2*br,
                                                    1.2*br,
                                                    self.border_color[:-2],
                                                    _hex2RGB(self.border_color)[3]/255.))
            
            # inner rectangle
            SVG.append('\t<rect x="{}" y="{}" width="{}" height="{}" '\
                       'rx="{}" ry="{}" fill="{}" '\
                       'fill-opacity="{:1.2f}" />'.format(self.center[0] - W/2,
                                                    self.center[1] - H/2,
                                                    W,
                                                    H,
                                                    br,
                                                    br,
                                                    self.background_color[:-2],
                                                    _hex2RGB(self.background_color)[3]/255.))
            
            
            # text
            SVG.append('\t<text x="{}" y="{}" text-anchor="middle" '\
                       'fill="{}" fill-opacity="{:1.2f}" font-size="{}" '\
                       'font-family="Ubuntu Light" dy=".3em">{}</text>'.format(self.center[0],
                                                            self.center[1],
                                                            self.font_color[:-2],
                                                            _hex2RGB(self.font_color)[3]/255.,
                                                            css_refs['globals']['font_tag']['font_size'],
                                                            self.tag
                                                            ))
            
            # if necessary, plot wrapper
            if self.wrapper is not None:
                SVG.append('\t<g>')
                SVG.append('\t\t<circle cx="{}" cy="{}" r="{}" stroke="{}" '\
                           'stroke-width="5" fill="{}" fill-opacity="{:1.2f}"/>'.format(self.center[0] - W/2,
                                                                self.center[1] - H/2,
                                                                css_refs['globals']['wrapper_radius'],
                                                                css_refs['wrappers'][self.wrapper_type]['border_color'][:-2],
                                                                css_refs['wrappers'][self.wrapper_type]['background_color'][:-2],
                                                                _hex2RGB(css_refs['wrappers'][self.wrapper_type]['background_color'])[3]/255.))
                
                # text
                SVG.append('\t\t<text x="{}" y="{}" text-anchor="middle" '\
                           'fill="black" font-size="30px" alignment-baseline="middle" '\
                           'font-family="Carlito-Regular" dy=".3em">{}</text>'.format(self.center[0] - W/2,
                                                                self.center[1] - H/2,
                                                                css_refs['wrappers'][self.wrapper_type]['tag']
                                                                ))
                SVG.append('\t</g>')
                                                            
            SVG.append('</g>')
           
            
        self.svg_tag = SVG
        return SVG
            

""" get model Svg """
def get_model_Svg(model,
                   filename=None,
                   verbose=False,
                   **kwargs):
    
    if (verbose):
        t0 = time.time()
        sys.stdout.write("Creating model_to_dot...")
        
    # Get model dot (optimal tags locations)
    ddot = model_to_dot(model, show_layer_names = True).create_plain().splitlines() # split linebreaks
    """ Decode """
    ddot = [d.decode('utf-8') for d in ddot]
    if (verbose):
        t1 = time.time()
        sys.stdout.write("%f (s)\n"%(t1-t0))
    
    """ dot parser """
    if (verbose):
        t2 = time.time()
        sys.stdout.write("Parsing dot data...")
        
    pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) "\
                         "([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) "\
                         '\"(.*?)\" (\w+) (\w+) (\w+) (\w+)')
    flds = ('id','x','y','width','height','old_tag','stroke', 'idk', 'col', 'col2')
    _int = lambda x: int(x)
    _float = lambda x: float(x)
    _str = lambda x: str(x)
    fun = (_int, _float, _float, _float, _float, _str, _str, _str, _str, _str)
    matches = [dict((x,f(y)) for f,x,y in zip(fun,flds, pattern.findall(d)[0])) \
               for d in ddot if 'node' in d]
    
    """ Match edges too """
    pattern = re.compile("edge (\d+) (\d+)")
    flds = ('id_from','id_to')
    _int = lambda x: int(x)
    _float = lambda x: float(x)
    _str = lambda x: str(x)
    fun = (_int, _int)
    edges = [dict((x,f(y)) for f,x,y in zip(fun,flds, pattern.findall(d)[0])) \
               for d in ddot if 'edge' in d]
    
    if (verbose):
        t3 = time.time()
        sys.stdout.write("%f (s)\n"%(t3-t2))


    def layer_parser(layer, x = 0, y = 0, **argsx):
        """ Init layer tag objects """
        return layerTag(layer = layer, center = (x,y), **argsx)

    """ Get layers info """
    import ctypes
    layers = [ctypes.cast(m['id'], ctypes.py_object).value for m in matches]
    old_sizes = {m['id']: (m['width'],m['height']) for m in matches}
    old_tags = {m['id']: m['old_tag'] for m in matches}
    layers = {m['id']: layer_parser(ly, **m) for m,ly in zip(matches,layers)}
    
    """ Let's recalculate the positions """
    zoom = np.array([(old_sizes[ly][0]/layers[ly].width * \
                          len(layers[ly].tag)/len(old_tags[ly]), \
                      old_sizes[ly][1]/layers[ly].height) \
            for ly in layers])
    zoom = np.max(zoom, axis = 0)
    
    zoom[1] *= .9
    
    """ Update positions for each tag item and Invert vertical direction 
        (for some reason ddot returns position in inverse y-axis)"""
    for ly in layers:
        layers[ly].center = (layers[ly].center[0]/zoom[0],
                             -layers[ly].center[1]/zoom[1])
        
    """ Let's now add extra dummy OUTPUT layers 
            A note here: See, we could actually go to the model and look 
            for the output layers using something like model.outputs, which
            would be easy piecey, if it wasn't because for some reason it looks
            like keras authors enjoy change this particular attribute from version
            to version. So in order to avoid unnecesary errors in the future, 
            instead of doing that we can actually look at 'edges' for those
            entries that only appear in the ID_TO field (and not in the ID_FROM).
            As tensorflow requires that no nodes are disconnected from the graph, any 
            node that fits this equirement will necessarily be an output layer
             
    """
    ids_from = [e['id_from'] for e in edges]
    ids_to = [e['id_to'] for e in edges]
    ids_outs = [e for e in ids_to if e not in ids_from]
    
    ids_to_outs = [[f for f,t in zip(ids_from,ids_to) if t == o] for o in ids_outs]
    outs_space = [np.min([layers[f].center[1]-layers[o].center[1] for f in t]) \
                  for t,o in zip(ids_to_outs,ids_outs)]
    
    #space = 80
    idmax = np.max([ids_from,ids_to])
    outTags = {idmax+1+i: layerTag(layer_type = 'OutputLayer', \
             center = (layers[e].center[0], layers[e].center[1] - outs_space[i]), \
             tag = 'Output: {}'.format(layers[e].layer.name)) \
        for i,e in enumerate(ids_outs)}
    
    layers.update(outTags)
    
    # update edges too
    [edges.append({'id_from':ids_outs[i],'id_to':e}) for i,e in enumerate(outTags)]
    
    """ Bounding Box calculation """
    border = 70
    """ Canvas Vertices """
    def _get_vertices(layers):
        
        xleft = np.min([layers[ly].center[0]-layers[ly].width/2 for ly in layers]) \
                    - border
        xright = np.max([layers[ly].center[0]+layers[ly].width/2 for ly in layers]) \
                    + border
        yup = np.min([layers[ly].center[1]-layers[ly].height/2 for ly in layers]) \
                    - border
        ydown = np.max([layers[ly].center[1]+layers[ly].height/2 for ly in layers]) \
                    + border
        
        return xleft, xright, yup, ydown
    
    xleft, xright, yup, ydown = _get_vertices(layers)
    
    """ Subtract offsets so drawing is set on a valid canvas position 
        (non negative)"""
    [setattr(layers[ly],'center',\
         (layers[ly].center[0] - xleft,
          layers[ly].center[1] - yup - ydown + border)) \
          for ly in layers]
    
    """ Now let's add the arrows """
    def _arrow_tag(edge):
        
        idfrom = edge['id_from']
        idto = edge['id_to']
        x0 = layers[idfrom].center[0]
        y0 = layers[idfrom].center[1] + layers[idfrom].height/2
        x1 = layers[idto].center[0]
        y1 = layers[idto].center[1] - layers[idto].height/2
        dy = css_refs['globals']['rounded_box_border_size']
        ey = 3
        arrow_head = 10 + ey
        
        bezierSvg = ['<g>']
        bezierSvg.append('<path stroke-width="{}" d="M {} {} C {} {}, {} {}, {} {}" '\
                            'stroke="black" fill="none" '\
                            'marker-end="url(#arrow)" />'.format(
                                css_refs['globals']['arrow_width'],
                                x0, y0 + dy,
                                x0, (y0 + y1)/2,
                                x1, (y0 + y1)/2,
                                x1, y1 - dy - arrow_head))
        # place shape text (if it changed from id0 to id1)
        if layers[idfrom].wrapper is None:
            ishfrom = layers[idfrom].layer.input_shape
            oshfrom = layers[idfrom].layer.output_shape
        else:
            ishfrom = layers[idfrom].wrapper.input_shape
            oshfrom = layers[idfrom].wrapper.output_shape
        if (ishfrom != oshfrom) or 'input' in layers[idfrom].layer_type.lower():
            dx = 10
            if x0 < x1:
                anchor = 'start'
            else:
                anchor = 'end'
                dx *= -1
            bezierSvg.append('<text x="{}" y="{}" text-anchor="{}" '\
                                'alignment-baseline="bottom" fill="#000000" '\
                                'font-size="20px" font-family="Ubuntu Light" '\
                                'dy=".3em">{}</text>\n'.format((x0 + x1)/2 + dx,
                                                               (y0 + y1)/2 ,
                                                               anchor,
                                                               str(oshfrom[1:])))
            
        bezierSvg.append('</g>')
        return bezierSvg
    
    arrows = ['\n'.join(['\t{}'.format(ee) for ee in _arrow_tag(e)]) for e in edges]
        
    """ Now get real bounding box """
    xleft, xright, yup, ydown = _get_vertices(layers)
    total_width = xright - xleft    
    total_height = ydown - yup + border
    
    """ Now build svg file """
    SvgTag = _SVG_init(total_width,total_height,x0=xleft,y0=yup,docname=filename)
    
    """ Rebuild svgs layer tags """
    SvgTag += '\n'.join(['\n'.join(['\t{}'.format(tt) for tt in layers[ly]._build_svg()]) \
               for ly in layers])
    
    """ Add arrows """
    SvgTag += '\n'.join(arrows)
        
    """ Close svg """
    SvgTag += '\m</svg>'
        
    """ And print file """
    svgFile = open(filename,"w", encoding="utf-8")
    svgFile.write(SvgTag)
    svgFile.close()
    
    return
    
