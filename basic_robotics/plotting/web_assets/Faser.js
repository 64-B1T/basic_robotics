import * as THREE from 'three';
// import { ColladaLoader } from './js/ColladaLoader.js';
// import { OrbitControls } from "./js/OrbitControls.js";
import { OrbitControls } from "https://threejs.org/examples/jsm/controls/OrbitControls.js";
import { ColladaLoader } from 'https://threejs.org/examples/jsm/loaders/ColladaLoader.js';
import { GLTFLoader } from 'https://threejs.org/examples/jsm/loaders/GLTFLoader.js';
import { STLLoader} from 'https://threejs.org/examples/jsm/loaders/STLLoader.js'
// import { Plotly} from 'https://cdn.plot.ly/plotly-latest.min.js';
// import { ImageLoader } from 'https://threejs.org/src/loaders/ImageLoader.js';

// Native three.js fat lines
import { Line2 } from 'https://threejs.org/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'https://threejs.org/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'https://threejs.org/examples/jsm/lines/LineGeometry.js';

// import { ArrowHelper } from 'https://threejs.org/src/helpers/ArrowHelper.js';

THREE.Object3D.DefaultUp = new THREE.Vector3(0,0,1);

//
// GLOBALS
//

// Rendered objects and HTML elements
var objects = {};
var htmls = {};
var timestamps = {};
var timeseries = {};
var lefttabs = [];
var righttabs = [];
var activelyUpdate = true; // Whether to call the API every frame or not
var sliderplay = [];
var jsoncategory = "";


// showingvisible: which divs are visible if showing is true
// hiddenvisible: which divs are visible if showing is false
var divStates = {"left":{"showing":true,
    "showingvisible":[document.getElementById("leftpanel"),
      document.getElementById("menuleftX")],
    "hiddenvisible":[document.getElementById("menuleft")]},
  "right":{"showing":true,
    "showingvisible":[document.getElementById("rightpanel"),
      document.getElementById("menurightX")],
    "hiddenvisible":[document.getElementById("menuright")]},
  "editor":{"showing":false,
    "showingvisible":[document.getElementById("editordiv")],
    "hiddenvisible":[]},
  "plot":{"showing":false,
    "showingvisible":[document.getElementById("plotdiv")],
    "hiddenvisible":[]},
};

// Scene information
var scene;
var renderer;
var camera;
var controls;
var sceneobjects = []; // For lights

//
// Scene, renderer, and camera
//
scene = new THREE.Scene();
scene["background"] = new THREE.Color( "rgb(117, 120, 123)"  );

renderer = new THREE.WebGLRenderer({ antialias: true });
// var renderwidth = document.getElementById( 'overgrid' ).clientWidth;
// var renderheight = document.getElementById( 'overgrid' ).clientHeight;
var renderwidth = window.innerWidth;
var renderheight = window.innerHeight;
renderer.setSize(renderwidth, renderheight);
// renderer.setViewport(0,0,50,renderheight);
// document.body.appendChild( renderer.domElement );
window.addEventListener( 'resize', onWindowResize, false );
document.getElementById( 'render' ).appendChild( renderer.domElement );

camera = new THREE.PerspectiveCamera( 75, renderwidth / renderheight, 0.1, 1000 );
camera.position.x = 5;
camera.position.y = 5;
camera.position.z = 5;

//
// Lighting
//

// Controller
controls = new OrbitControls( camera, renderer.domElement );
controls.update();

var activate = document.getElementById( "activate" );
activate.addEventListener( 'click', function () {
  if (activelyUpdate) {
    activelyUpdate = false;
    this.innerHTML = "Activate";
  } else {
    activelyUpdate = true;
    this.innerHTML = "Deactivate";
  }
});

// UI
document.getElementById("menuleft").addEventListener( 'click', function () {toggleDiv("left");});
document.getElementById("menuleftX").addEventListener( 'click', function () {toggleDiv("left");});
document.getElementById("menuright").addEventListener( 'click', function () {toggleDiv("right");});
document.getElementById("menurightX").addEventListener( 'click', function () {toggleDiv("right");});
document.getElementById("showplot").addEventListener( 'click', function () {toggleDiv("plot");});
document.getElementById("showeditor").addEventListener( 'click', function () {toggleDiv("editor");});

// Limit JSON calls to specific API only when URL ends in page.html/category
var spliturl = location.toString().split("?");
if (spliturl.length > 1) {
  jsoncategory = "Category=" + spliturl[spliturl.length-1] + "&";
}


// TESTING

// var trace1 = {
//   x:['2020-10-04', '2021-11-04', '2023-12-04'],
//   y: [90, 40, 60],
//   type: 'scatter'
// };

// var datap = [trace1];

// var layoutp = {
//   title: 'Scroll and Zoom',
//   showlegend: false
// };

// Plotly.newPlot('plotinner', datap, layoutp, {scrollZoom: true});

// END TESTING


// getUpdate();

animate();
update();
// setInterval(update,16.66667)

function toggleDiv(divid) {
  // Hide everything else if this is a mobile display
  if (window.getComputedStyle(document.getElementById("faserlogomobilediv"),null).getPropertyValue("display") == "block") {
    for (const divname in divStates) {
      if (divid == divname) {continue;}
      divStates[divname].showing = false;
      for (const shows in divStates[divname].showingvisible) {
        divStates[divname].showingvisible[shows].style.display = "none";
      }
      for (const hides in divStates[divname].hiddenvisible) {
        divStates[divname].hiddenvisible[hides].style.display = "block";
      }
    }
  }
  if (divStates[divid].showing == true) {
    divStates[divid].showing = false;
    for (const shows in divStates[divid].showingvisible) {
      divStates[divid].showingvisible[shows].style.display = "none";
    }
    for (const hides in divStates[divid].hiddenvisible) {
      divStates[divid].hiddenvisible[hides].style.display = "block";
    }
  } else {
    divStates[divid].showing = true;
    for (const shows in divStates[divid].showingvisible) {
      divStates[divid].showingvisible[shows].style.display = "block";
    }
    for (const hides in divStates[divid].hiddenvisible) {
      divStates[divid].hiddenvisible[hides].style.display = "none";
    }
  }
}

function onWindowResize() {

  // var renderwidth = document.getElementById( 'overgrid' ).clientWidth;
  // var renderheight = document.getElementById( 'overgrid' ).clientHeight;
  var renderwidth = window.innerWidth;
  var renderheight = window.innerHeight;
  // console.log(renderheight);
  renderer.setSize(renderwidth, renderheight);

  camera.aspect = renderwidth / renderheight;
  camera.updateProjectionMatrix();
  renderer.setSize( renderwidth, renderheight );
}

function animate() {
  requestAnimationFrame( animate );

  for (const slid in sliderplay) {
    if (sliderplay[slid].playing == true) {
      var speedstep = parseFloat(sliderplay[slid].speed.value) / 60.0;
      if (Math.abs(speedstep) < sliderplay[slid].slider.step) {
        speedstep = Math.sign(speedstep) * sliderplay[slid].slider.step;
        sliderplay[slid].speed.value = speedstep * 60.0;
      }
      var newval = parseFloat(sliderplay[slid].slider.value) + speedstep;
        // *(sliderplay[slid].slider.max - sliderplay[slid].slider.min);
      if (newval > sliderplay[slid].slider.max) {newval = sliderplay[slid].slider.min;}

      // Update slider and text box
      sliderplay[slid].slider.value = newval;
      sliderplay[slid].valTB.value = newval;

      // Update time series
      PickTimeSeries(sliderplay[slid].callerbase,newval);
    }
  }

  // getUpdate();
  controls.update();
  renderer.render( scene, camera );
};

function update() {
  requestAnimationFrame( update );
  if (activelyUpdate) {
    getUpdate();
  }
}

function getUpdate() {
  fetch('/api/json?' + jsoncategory + 'Latest=1')
  .then(response => response.json())
  .then(data => handleJSON(data))
}

function putUpdate(opts) {
  var xv = 0.;
  var yv = 0.;
  if ('xvalue' in opts) {
    xv = opts.xvalue;
  }
  if ('yvalue' in opts) {
    yv = opts.yvalue;
  }
}

function PutAlphanumericControl(id) {
  const thevalue = document.getElementById(id).value;
  const data = {"Key":"Control:"+id,
    "Value":(isNaN(thevalue)) ? thevalue : parseFloat(thevalue),
    "UnixTime":Date.now()/1000};
  fetch('/api/json',{
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
}

function PutBooleanControl(id) {
  const data = {"Key":"Control:"+id,"Value":document.getElementById(id).checked, "UnixTime":Date.now()/1000};
  fetch('/api/json',{
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
}

function PutLabelControl(id) {
  const data = {"Key":"Control:"+id,"Value":document.getElementById(id).innerHTML, "UnixTime":Date.now()/1000};
  fetch('/api/json',{
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
}

function PutClickControl(id) {
  const data = {"Key":"Control:"+id,"Value":true, "UnixTime":Date.now()/1000};
  fetch('/api/json',{
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
}

function GetControl(id) {
  fetch(id)
  .then(response => response.json())
  .then(data => handleJSON(data))
}

function handleJSON(data) {
  for (const key in data) {
    // Update if this has a time stamp that is expired, or has no time stamp
    var tohandle = false;
    if ("UnixTime" in data[key]) {
      if (!(key in timestamps) || data[key]["UnixTime"] != timestamps[key]) {
        timestamps[key] = data[key]["UnixTime"];
        tohandle = true;
      }
    } else {
      tohandle = true;
    }

    if (tohandle == true) {
      if ("Scene" in data[key]) {
        handleScene(data[key]);
      }
      // This is an object with a transform
      if ("Matrix" in data[key]) {
        handleObject(data[key]);
      }
      if ("Billboard" in data[key]) {
        handleBillboard(data[key]);
      }
      // Line segment
      else if ("Segments" in data[key]) {
        handleLine(data[key]);
      }
      // Control definitions
      else if ("ControlGroups" in data[key]) {
        handleControls(data[key]);
      }
      else if ("ControlUpdates" in data[key]) {
        handleControlUpdates(data[key]);
      }
      else if ("Times" in data[key]) {
        // Just store the message
        timeseries[key] = data[key];
      }
    }
    if (key in objects) {objects[key].stillexists = true;}
    if (key in htmls) {htmls[key].stillexists = true;}
  }
  removeDeadObjects();
  removeDeadHTMLs();
}

function handleScene(data) {
  for (var i=0; i < sceneobjects.length; i++) {
    scene.remove(sceneobjects[i]);
  }
  sceneobjects = [];

  scene.background = new THREE.Color( data["Background"] );

  if (data["Fog"]["Type"] == "Exp") {
    scene.fog = new THREE.FogExp2( data["Fog"]["Color"] , data["Fog"]["Density"] );
  }

  if (data["Camera"]["Type"] == "Perspective") {
    camera = new THREE.PerspectiveCamera( data["Camera"]["FOV"], window.innerWidth / window.innerHeight,
      data["Camera"]["Near"], data["Camera"]["Far"] );
    camera.position.set(data["Camera"]["Position"][0],
      data["Camera"]["Position"][1],data["Camera"]["Position"][2]);
    controls = new OrbitControls( camera, renderer.domElement );
    controls.update();
  }

  for (var i = 0; i < data["Lights"].length; i++) {
    var light = data["Lights"][i];
    if (light["Type"] == "Ambient") {
      var lig = new THREE.AmbientLight(light["Color"],light["Intensity"]);
      scene.add(lig);
      sceneobjects.push(lig);
    }
    if (light["Type"] == "Directional") {
      var lig = new THREE.DirectionalLight(light["Color"],light["Intensity"]);
      lig.position.set(light["Position"][0],light["Position"][1],light["Position"][2]).normalize();
      scene.add(lig);
      sceneobjects.push(lig);
    }
    if (light["Type"] == "Point") {
      var lig = new THREE.PointLight(light["Color"],light["Intensity"],light["Distance"],light["Decay"]);
      lig.position.set(light["Position"][0],light["Position"][1],light["Position"][2]).normalize();
      scene.add(lig);
      sceneobjects.push(lig);
    }
  }
}

function handleObject(data) {
  if ("Key" in data) {
    if (!(data["Key"] in objects)) {
      // Load a Collada file
      if ("File" in data) {
        var fnamesplit = data["File"].split(".");
        if (fnamesplit[fnamesplit.length - 1] === "dae") {
          objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
          var newobj;
          var loader = new ColladaLoader();
          console.log("loading");
          loader.load('http://localhost:5000/' + data["File"], function ( collada ) {
            // This is asynchronous, so without care it might be called repeatedly
            console.log(collada);
            newobj = collada.scene;
            newobj.object = null;

            scene.add(newobj);

            objects[data["Key"]].object = newobj;
            // objects[data["Key"]].stillexists = true;
            replaceMatrix(newobj,data.Matrix);
          },(xhr) => {
            console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
          },
          (error) => {
              console.log(error)
          });
        } else if (fnamesplit[fnamesplit.length - 1] === "glb") {
          objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
          var newobj;
          var loader = new GLTFLoader();
          loader.load(data["File"], function ( gltf ) {
            // This is asynchronous, so without care it might be called repeatedly
            newobj = gltf.scene;
            newobj.object = null;
            // if ("Wireframe" in data) {
            //   const wireframe = new THREE.WireframeGeometry( newobj.geometry );
            //   newobj = new THREE.LineSegments( wireframe );
            //   newobj.material.depthTest = false;
            //   newobj.material.opacity = 0.25;
            //   newobj.material.transparent = true;
            // }
            if ("Scale" in data) {
              for (var i in newobj.children) {
                newobj.children[i].geometry.scale(data["Scale"][0],data["Scale"][1],data["Scale"][2]);
              }
            }
            
            scene.add(newobj);
            
            objects[data["Key"]].object = newobj;
            // objects[data["Key"]].stillexists = true;
            replaceMatrix(newobj,data.Matrix);
          });
        } else if (fnamesplit[fnamesplit.length - 1] === "stl") {
          objects[data["Key"]] = {};
          console.log(data["File"]);
          var newobj;

          var loader = new STLLoader();

          const material = new THREE.MeshPhysicalMaterial({
            color: 0xb2ffc8,
            metalness: 0.25,
            roughness: 0.1,
            opacity: 1.0,
            transparent: true,
            transmission: 0.99,
            clearcoat: 1.0,
            clearcoatRoughness: 0.25
          });
        
          
          loader.load(data["File"], function ( geometry  ) {
            console.log(geometry);
            console.log('loading');
            newobj = geometry.scene;
            newobj = new THREE.Mesh( geometry , material);
            newobj.position.set(0, 0, 0);
            newobj.rotation.set(0, 0, 0);
            scene.add( newobj );

            objects[data["Key"]].object = newobj;
            console.log('loaded');
            // objects[data["Key"]].stillexists = true;
            replaceMatrix(newobj,data.Matrix);
          });
          
        }
      }
      // Other object creation methods go here
      else if ("Frame" in data) {
        objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
        var newobj = new THREE.Group();
        var scale = 1;
        if ("Scale" in data) {
          scale = data["Scale"];
        }
        var xmesh = createLine([[0.,0.,0.],[scale,0.,0.]],
          {color:[1.,0.,0.],lineWidth:0.0025,sizeAttenuation:1});
        newobj.add(xmesh);
        var ymesh = createLine([[0.,0.,0.],[0.,scale,0.]],
          {color:[0.,1.,0.],lineWidth:0.0025,sizeAttenuation:1});
        newobj.add(ymesh);
        var zmesh = createLine([[0.,0.,0.],[0.,0.,scale]],
          {color:[0.,0.,1.],lineWidth:0.0025,sizeAttenuation:1});
        newobj.add(zmesh);
        scene.add(newobj);
        objects[data["Key"]].object = newobj;
        // objects[data["Key"]].stillexists = true;
        replaceMatrix(newobj,data.Matrix);
      } else if ("Primitive" in data) {
        // NOTE/TODO/TO DO: Add more primitives as needed.
        console.log('making primitive');
        objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
        let newobj;

        // User provides the material already in parameters format
        const material = new THREE.MeshStandardMaterial(data["Material"]);

        if (data["Primitive"] == "Box") {
          const geometry = new THREE.BoxGeometry(1,1,1);
          geometry.scale(data["Scale"][0], data["Scale"][1], data["Scale"][2]);
          newobj = new THREE.Mesh( geometry, material );
        } else if (data["Primitive"] == "Ellipsoid") {
          const geometry = new THREE.SphereGeometry(1,20,20);
          geometry.scale(data["Scale"][0], data["Scale"][1], data["Scale"][2]);
          newobj = new THREE.Mesh( geometry, material );
        } else if (data["Primitive"] == "Cylinder") {
          const geometry = new THREE.CylinderGeometry(0.5,0.5,1,20);
          geometry.scale(data["Scale"][0], data["Scale"][1], data["Scale"][2]);
          newobj = new THREE.Mesh( geometry, material );
        } else if (data["Primitive"] == "Torus") {
        } else if (data["Primitive"] == "Tube") {
        }

        scene.add(newobj);
        objects[data["Key"]].object = newobj;
        // objects[data["Key"]].stillexists = true;
        replaceMatrix(newobj,data.Matrix);
      }
    } else {
      // Update the information
      var updobj = objects[data["Key"]];
      // Make sure this isn't still the placeholder
      if (Object.keys(updobj).length !== 0) {
        // updobj.stillexists = true;
        // Make sure data exists
        if ("Matrix" in data) {
          replaceMatrix(updobj.object,data.Matrix);
        }
      }
    }
  }
}

function handleBillboard(data) {
  if ("Key" in data) {
    if (!(data["Key"] in objects)) {
      objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
      var spriteMap;
      var spriteMaterial;
      var scale;
      if ("Image" in data) {
        spriteMap = new THREE.TextureLoader().load( data["Image"] );
        spriteMaterial = new THREE.SpriteMaterial( { map: spriteMap } );
        scale = [data["Scale"][0],data["Scale"][1],1];
        // console.log(spriteMap);
        // scale = [1.,1.,1.];
        // scale = [data["Scale"]*spriteMap.image.width/spriteMap.image.height,data["Scale"],1];
      } else if ("Text" in data) {
        var context = document.createElement('canvas').getContext('2d');
        const fontsize = 48;
        context.font = `${fontsize}px Verdana`;
        var lines = data["Text"].split('\n');
        var width = 0;
        var height = fontsize*lines.length;
        for (var i = 0; i < lines.length; i++) {
          var lwidth = context.measureText(lines[i]).width;
          if (lwidth > width) {width = lwidth;}
        }
        context.canvas.width = width;
        context.canvas.height = height;
        context.font = `${fontsize}px Verdana`;
        context.textBaseline = 'top';
        context.fillStyle = data["Background"];
        context.fillRect(0, 0, width, height);
        context.fillStyle = data["Color"];
        for (var i = 0; i < lines.length; i++) {
          context.fillText(lines[i], 0, i*fontsize);
        }
        spriteMap = new THREE.CanvasTexture(context.canvas);
        spriteMaterial = new THREE.SpriteMaterial( { map: spriteMap } );
        scale = [data["Scale"]*width/height*lines.length,data["Scale"]*lines.length,1];
      }
      var sprite = new THREE.Sprite( spriteMaterial );
      sprite.position.x = data["Position"][0];
      sprite.position.y = data["Position"][1];
      sprite.position.z = data["Position"][2];
      sprite.center.set(0.5,0.5);
      sprite.scale.set(scale[0],scale[1],scale[2]);
      scene.add( sprite );
      objects[data["Key"]].object = sprite;
    }
    else {
      removeObject(data["Key"]);
      handleBillboard(data);
      // // Update the information
      // var updobj = objects[data["Key"]];
      // // Make sure this isn't still the placeholder
      // if (Object.keys(updobj).length !== 0) {
      // 	// updobj.stillexists = true;
      // 	// Make sure data exists
      // 	if ("Matrix" in data) {
      // 		replaceMatrix(updobj.object,data.Matrix);
      // 	}
      // }
    }
  }
}

function handleLine(data) {

  if ("Key" in data) {
    if (!(data["Key"] in objects)) {
      objects[data["Key"]] = {}; // Placeholder to prevent multiple loads
      var mesh = createLine(data["Segments"],data["LineParameters"]);
      scene.add( mesh );
      objects[data["Key"]].object = mesh;
      objects[data["Key"]].stillexists = true;  
    }
    else {
      removeObject(data["Key"]);
      handleLine(data);
    }
  }

}

function handleControls(data) {
  if ("Key" in data) {
    if (!(data["Key"] in htmls)) {
      htmls[data["Key"]] = {children:[]};

      if ("ControlGroups" in data) {
        createGroupMenu(data,htmls[data["Key"]].children,false);

        for (const cg of data["ControlGroups"]) {
          createGroup(cg,htmls[data["Key"]].children,false);
        }

        if (lefttabs.length > 0) lefttabs[0].style.display = "grid";
      }

      if ("DataGroups" in data) {
        createGroupMenu(data,htmls[data["Key"]].children,true);

        for (const dg of data["DataGroups"]) {
          createGroup(dg,htmls[data["Key"]].children,true);
        }

        if (righttabs.length > 0) righttabs[0].style.display = "grid";
      }
    }
    // htmls[data["Key"]].stillexists = true;
  }
}

function handleControlUpdates(data) {
  if ("Key" in data) {
    // if (data["Key"] in timestamps && data["UnixTime"] == timestamps[data["Key"]]) {
    // 	return;
    // }
    for (const idkey in data["ControlUpdates"]) {
      var val = document.getElementById(idkey);
      if (val != null) {
        val.value = data["ControlUpdates"][idkey];
      }
      var slider = document.getElementById(idkey + " Slider");
      if (slider != null) {
        slider.value = data["ControlUpdates"][idkey];
      }
      var tb = document.getElementById(idkey + " Text");
      if (tb != null) {
        tb.innerHTML = data["ControlUpdates"][idkey];
      }
    }
  }
}

function removeObject(key) {
  if (objects[key].object == undefined)
  {
    delete objects[key];
    return;
  }
  if (objects[key].object.geometry !== undefined) {
    objects[key].object.geometry.dispose();
  }
  if (objects[key].object.material !== undefined) {
    objects[key].object.material.dispose();
  }
  scene.remove(objects[key].object);
  delete objects[key];
}

function removeDeadObjects() {
  for (const key in objects) {
    if ("stillexists" in objects[key]) {
      if (!objects[key].stillexists) {
        removeObject(key);
      } else {
        // Default to false so that if the next update doesn't have it, it's deleted then
        objects[key].stillexists = false;
      }
    }
  }
}

function removeHTML(key) {
  for (var i=0; i < htmls[key].children.length; i++) {
    htmls[key].children[i].remove();
  }
  delete htmls[key];
  // const ind = lefttabs.indexOf(key);
  // if (ind > -1) {lefttabs.splice(ind,1);}
  // ind = righttabs.indexOf(key);
  // if (ind > -1) {righttabs.splice(ind,1);}
}

function removeDeadHTMLs() {
  // TO DO: Code currently set up to have a single HTML key at a time
  // Updating control setup requires deleting and creating again
  var htmldeleted = false;
  for (const key in htmls) {
    if ("stillexists" in htmls[key]) {
      if (!htmls[key].stillexists) {
        removeHTML(key);
        htmldeleted = true;
      } else {
        // Default to false so that if the next update doesn't have it, it's deleted then
        htmls[key].stillexists = false;
      }
    }
  }
  if (htmldeleted) {
    // TO DO: Fix these lines if multiple control groups
    lefttabs = [];
    righttabs = [];
    sliderplay = [];
  }
}

function replaceMatrix ( obj, matrix ) {
  // It turns out Object3D doesn't actually do this on its own
  if (obj != null)
  {
    var m = new THREE.Matrix4();
    m.fromArray(matrix);
    if ( obj.matrixAutoUpdate ) obj.updateMatrix();
    obj.matrix = m;
    obj.matrix.decompose( obj.position, obj.quaternion, obj.scale );
  }
  
}

function createLine(points,params) {
  if ("Arrow" in params) {
    // Javascript does not have much in the way of basic math, so rely on the user to do this.
    const p = new THREE.Vector3();
    // p.fromArray(points[0]);
    p.fromArray(params["ArrowBase"]);
    const dir = new THREE.Vector3();
    // dir.fromArray(points[1]-points[0]);
    dir.fromArray(params["ArrowDirection"]);
    // const length = Math.hypot(dir.x,dir.y,dir.z);
    // dir = dir / length;
    const length = params["ArrowLength"];
    const color = new THREE.Color(params["color"][0],params["color"][1],params["color"][2]);
    const colorhex = color.getHex();
    const arrow = new THREE.ArrowHelper(dir, p, length, colorhex);
    // arrow.renderOrder = 999;
    // arrow.onBeforeRender = function( renderer ) { renderer.clearDepth(); };
    return arrow;
  } else {
    const positions = [];
    const colors = [];
    for (const point of points) {
      positions.push(point[0],point[1],point[2]);
      colors.push(params["color"][0], params["color"][1], params["color"][2]);
    }

    const geometry = new LineGeometry();
    geometry.setPositions( positions );
    geometry.setColors( colors );
    const matLine = new LineMaterial( {
      color: 0xffffff,
      linewidth: params["lineWidth"], // in percent of viewframe
      vertexColors: true,
      dashed: false
    } );

    const line = new Line2( geometry, matLine );
    line.computeLineDistances();
    line.scale.set( 1, 1, 1 );
    line.renderOrder = 999;
    line.onBeforeRender = function( renderer ) { renderer.clearDepth(); };
    return line;
  }
}

// function createLineOld(points,params) {
//   // console.log(points);
//   var geometry = new THREE.Geometry();
//   for (const point of points) {
//     var p = new THREE.Vector3();
//     p.fromArray(point);
//     geometry.vertices.push(p);
//   }

//   var line = new MeshLine();
//   line.setGeometry( geometry );
//   var material = new MeshLineMaterial(params);
//   var mesh = new THREE.Mesh( line.geometry, material ); // this syntax could definitely be improved!

//   // These lines force the lines to render on top of the rest
//   mesh.renderOrder = 999;
//   mesh.onBeforeRender = function( renderer ) { renderer.clearDepth(); };
//   return mesh;
// }

function createGroupMenu(data,children,isRight) {
  var panelname = 'leftpanel';
  var dropside = 'fvdrop-content';
  var groupname = "ControlGroups";
  var groupdisp = "Control Tabs";
  if (isRight) {
    panelname = 'rightpanel';
    dropside = 'fvdrop-content-right';
    groupname = "DataGroups";
    groupdisp = "Data Tabs";
  }
  // Create the div and its children that support the menu
  var egdiv = document.createElement('div');
  egdiv.setAttribute('class','elementGrid');
  // egdiv.setAttribute('id',data["Key"]);
  document.getElementById(panelname).appendChild(egdiv);
  children.push(egdiv);
  var fediv = document.createElement('div');
  fediv.setAttribute('class','fvelement');
  egdiv.appendChild(fediv);
  var ddiv = document.createElement('div');
  ddiv.setAttribute('class','fvdrop');
  fediv.appendChild(ddiv);
  ddiv.innerHTML = groupdisp;
  // var dbutton = document.createElement('button');
  // dbutton.setAttribute('class','fvdropbutton');
  // dbutton.innerHTML = groupname;
  // ddiv.appendChild(dbutton);
  var dcdiv = document.createElement('div');
  dcdiv.setAttribute('class',dropside);
  ddiv.appendChild(dcdiv);
  for (const cg of data[groupname]) {
    var egdiv = document.createElement('div');
    egdiv.setAttribute('class','elementGrid');
    dcdiv.appendChild(egdiv);
    var fediv = document.createElement('div');
    fediv.setAttribute('class','fvelement');
    egdiv.appendChild(fediv);
    var dcbutton = document.createElement('button');
    dcbutton.setAttribute('class','fvbutton');
    dcbutton.innerHTML = cg["GroupName"];
    dcbutton.setAttribute('id',cg["GroupName"] + " Tab Button");
    fediv.appendChild(dcbutton);
  }
}

function createGroup(cg,children,isRight) {
  var panelname = 'leftpanel';
  var dropside = 'fvdrop-content';
  if (isRight) {
    panelname = 'rightpanel';
    dropside = 'fvdrop-content-right';
  }

  var tab = document.createElement('div');
  tab.setAttribute('class','tabGrid');
  tab.setAttribute('id',cg["GroupName"] + " Tab");
  // tab.style.display = "grid";
  document.getElementById(panelname).appendChild(tab);
  children.push(tab);
  var dcbutton = document.getElementById(cg["GroupName"] + " Tab Button");
  if (isRight) {
    righttabs.push(tab);
    dcbutton.addEventListener( 'click',
      generateTabButtonCallback({show:tab,tablist:righttabs}));
  } else {
    lefttabs.push(tab);
    dcbutton.addEventListener( 'click',
      generateTabButtonCallback({show:tab,tablist:lefttabs}));
  }

  for (const item of cg["Items"]) {
    if ("Label" in item) {
      var egdiv = document.createElement('div');
      egdiv.setAttribute('class','elementGrid');
      tab.appendChild(egdiv);
      var fediv = document.createElement('div');
      fediv.setAttribute('class','fvelement');
      egdiv.appendChild(fediv);
      var label = document.createElement('label');
      label.setAttribute('class','fvlabel');
      label.innerHTML = item["Label"];
      fediv.appendChild(label);
    } else if ("Value" in item || "TimeSeries" in item) {
      var valID;
      if ("Value" in item) {valID = item["Value"];}
      else {valID = item["TimeSeries"]}
      var vgdiv = document.createElement('div');
      vgdiv.setAttribute('class','valGroupGrid');
      tab.appendChild(vgdiv);

      // Label
      var vglabel = document.createElement('div');
      vglabel.setAttribute('class','vglabel');
      vgdiv.appendChild(vglabel);
      var label = document.createElement('label');
      label.setAttribute('class','fvlabel');
      label.innerHTML = valID;
      vglabel.appendChild(label);

      // Textbox
      var vgtb = document.createElement('div');
      vgtb.setAttribute('class','vgtextbox');
      vgdiv.appendChild(vgtb);
      var textin = document.createElement('INPUT');
      textin.setAttribute('type','text');
      textin.value = item["Init"];
      textin.setAttribute('class','fvtextbox');
      textin.setAttribute('id',valID);
      vgtb.appendChild(textin);

      // If a range is provided, add a slider
      if ("MinMaxStep" in item) {
        var egdiv = document.createElement('div');
        egdiv.setAttribute('class','elementGrid');
        tab.appendChild(egdiv);
        var fediv = document.createElement('div');
        fediv.setAttribute('class','fvelement');
        egdiv.appendChild(fediv);
        var slider = document.createElement('INPUT');
        slider.setAttribute('type','range');
        slider.min = item["MinMaxStep"][0];
        slider.max = item["MinMaxStep"][1];
        slider.step = item["MinMaxStep"][2];
        slider.value = item["Init"];
        slider.setAttribute('class','fvslider');
        slider.setAttribute('id',valID+" Slider");
        textin.addEventListener('keyup',
          generateTBCallback({caller:valID,updates:[valID+" Slider"],
          send:("Send" in item) ? item["Send"] : [], series:("TimeSeries" in item),
          callerbase:valID}));
        slider.oninput = generateSliderCallback({caller:valID+" Slider",updates:[valID],
          send:("Send" in item) ? item["Send"] : [], series:("TimeSeries" in item),
          callerbase:valID});
        fediv.appendChild(slider);
        if ("TimeSeries" in item) {
          // Make a play button
          var vgdiv = document.createElement('div');
          vgdiv.setAttribute('class','valGroupGrid');
          tab.appendChild(vgdiv);

          // Speed setting
          var vgtb = document.createElement('div');
          vgtb.setAttribute('class','vglefttb');
          vgdiv.appendChild(vgtb);
          var textin2 = document.createElement('INPUT');
          textin2.setAttribute('type','text');
          textin2.value = 1;
          textin2.setAttribute('class','fvtextbox');
          textin2.setAttribute('id',valID);
          vgtb.appendChild(textin2);

          // Play/stop button
          var vgbtn = document.createElement('div');
          vgbtn.setAttribute('class','vgrightbutton');
          vgdiv.appendChild(vgbtn);
          var button = document.createElement('button');
          button.innerHTML = "Play";
          button.setAttribute('class','fvbutton');
          vgbtn.appendChild(button);

          var slid = {"slider":slider, "speed":textin2, "callerbase":valID, "playing":false,
            "min":item["MinMaxStep"][0], "max":item["MinMaxStep"][1],  "step":item["MinMaxStep"][2],
            "button":button, "valTB":textin};
          sliderplay.push(slid);
          button.addEventListener( 'click', generatePlayButtonCallback(sliderplay.length-1));
          // button.addEventListener( 'click',
          //   generateButtonCallback({numsend:("Send" in item) ? item["Send"] : [],
          //   boolsend:[],clicksend:[],get:("Get" in item) ? item["Get"] : []}));
          // var newval = slid.slider.value + slid.speed.value/60.0;
          //
          // // Update slider
          // slid.slider.value = newval;
          //
          // // Update time series
          // PickTimeSeries(slid.callerbase,newval);
        }
      } else {
        textin.addEventListener('keyup',
          generateTBCallback({caller:valID,updates:[],
          send:("Send" in item) ? item["Send"] : [], series:("TimeSeries" in item),
          callerbase:valID}));
      }
    } else if ("Button" in item) {
      var egdiv = document.createElement('div');
      egdiv.setAttribute('class','elementGrid');
      tab.appendChild(egdiv);
      var fediv = document.createElement('div');
      fediv.setAttribute('class','fvelement');
      egdiv.appendChild(fediv);
      var button = document.createElement('button');
      button.innerHTML = item["Button"];
      button.setAttribute('class','fvbutton');
      // button.setAttribute('id',vg["Button"]["ID"]);
      fediv.appendChild(button);
      button.addEventListener( 'click',
        generateButtonCallback({numsend:("Send" in item) ? item["Send"] : [],
        boolsend:("Send Checks" in item) ? item["Send Checks"] : [],
        labelsend:("Send Labels" in item) ? item["Send Labels"] : [],
        clicksend:("Send Clicks" in item) ? item["Send Clicks"] : [],
        get:("Get" in item) ? item["Get"] : []}));
    } else if ("Text" in item) {
      var egdiv = document.createElement('div');
      egdiv.setAttribute('class','elementGrid');
      tab.appendChild(egdiv);
      var fediv = document.createElement('div');
      fediv.setAttribute('class','fvleftelement');
      egdiv.appendChild(fediv);
      var textbox = document.createElement('label');
      textbox.setAttribute('class','fvdatalabel');
      textbox.setAttribute('id',item["ID"]+" Text");
      textbox.innerHTML = item["Text"];
      fediv.appendChild(textbox);
    } else if ("Select" in item) {

      // Create the div and its children that support the menu
      var egdiv = document.createElement('div');
      egdiv.setAttribute('class','elementGrid');
      tab.appendChild(egdiv);
      children.push(egdiv);
      var fediv = document.createElement('div');
      fediv.setAttribute('class','fvelement');
      egdiv.appendChild(fediv);
      var ddiv = document.createElement('div');
      ddiv.setAttribute('class','fvdrop');
      fediv.appendChild(ddiv);
      // ddiv.innerHTML = item["Options"][item["Selected"]];
      var dcdiv = document.createElement('div');
      dcdiv.setAttribute('class',dropside);
      ddiv.appendChild(dcdiv);
      // ddiv.setAttribute('id',item["Select"]);
      var label = document.createElement('label');
      label.setAttribute('class','fvlabel');
      label.innerHTML = item["Options"][item["Selected"]];
      ddiv.appendChild(label);
      label.setAttribute('id',item["Select"]);
      // label.addEventListener( 'change',
      //   generateButtonCallback({numsend:("Send" in item) ? item["Send"] : [],
      //   boolsend:("Send Checks" in item) ? item["Send Checks"] : [],
      //   labelsend:("Send Labels" in item) ? item["Send Labels"] : [],
      //   clicksend:("Send Clicks" in item) ? item["Send Clicks"] : [],
      //   get:("Get" in item) ? item["Get"] : []}));
      for (const cg of item["Options"]) {
        var egdiv = document.createElement('div');
        egdiv.setAttribute('class','elementGrid');
        dcdiv.appendChild(egdiv);
        var fediv = document.createElement('div');
        fediv.setAttribute('class','fvelement');
        egdiv.appendChild(fediv);
        var dcbutton = document.createElement('button');
        dcbutton.setAttribute('class','fvbutton');
        dcbutton.innerHTML = cg;
        dcbutton.setAttribute('id',item["Select"] + ":" + cg);
        fediv.appendChild(dcbutton);
        dcbutton.addEventListener('click',generateSelectButtonCallback({choice:cg,main:item["Select"]}));
        dcbutton.addEventListener( 'click',
          generateButtonCallback({numsend:("Send" in item) ? item["Send"] : [],
          boolsend:("Send Checks" in item) ? item["Send Checks"] : [],
          labelsend:("Send Labels" in item) ? item["Send Labels"] : [],
          clicksend:("Send Clicks" in item) ? item["Send Clicks"] : [],
          get:("Get" in item) ? item["Get"] : []}));
      }

    } else if ("Checkbox" in item) {
      var vgdiv = document.createElement('div');
      vgdiv.setAttribute('class','valGroupGrid');
      tab.appendChild(vgdiv);

      // Label
      var vglabel = document.createElement('div');
      vglabel.setAttribute('class','vglabel');
      vgdiv.appendChild(vglabel);
      var label = document.createElement('label');
      label.setAttribute('class','fvlabel');
      label.innerHTML = item["Checkbox"];
      vglabel.appendChild(label);

      // Checkbox
      var vgtb = document.createElement('div');
      vgtb.setAttribute('class','vgtextbox');
      vgdiv.appendChild(vgtb);
      var check = document.createElement('INPUT');
      check.setAttribute('type','checkbox');
      check.checked = item["Checked"];
      check.setAttribute('id',item["Checkbox"]);
      vgtb.appendChild(check);
      check.addEventListener( 'change',
        generateButtonCallback({numsend:("Send" in item) ? item["Send"] : [],
        boolsend:("Send Checks" in item) ? item["Send Checks"] : [],
        labelsend:("Send Labels" in item) ? item["Send Labels"] : [],
        clicksend:("Send Clicks" in item) ? item["Send Clicks"] : [],
        get:("Get" in item) ? item["Get"] : []}));
    }
  }
}

function generateTabButtonCallback(arg) {
  return function() {
    for (const tab of arg.tablist) {
      tab.style.display = "none";
    }
    arg.show.style.display = "grid";
  };
}

function generateTBCallback(arg) {
  return function(e) {
    if (e.keyCode === 13) {
      var val = document.getElementById(arg.caller).value;
      for (const upid of arg.updates) {
        document.getElementById(upid).value = val;
      }
      for (const sendid of arg.send) {
        PutAlphanumericControl(sendid);
      }
      if (arg.series) {
        PickTimeSeries(arg.callerbase,val);
      }
    }
  };
}

function generateSliderCallback(arg) {
  return function() {
    var val = document.getElementById(arg.caller).value;
    for (const upid of arg.updates) {
      document.getElementById(upid).value = val;
    }
    for (const sendid of arg.send) {
      PutAlphanumericControl(sendid);
    }
    if (arg.series) {
      PickTimeSeries(arg.callerbase,val);
    }
  };
}

function generateSelectButtonCallback(arg) {
  return function() {
    document.getElementById(arg.main).innerHTML = arg.choice;
  }
}

function generateButtonCallback(arg) {
  return function() {
    for (const sendid of arg.numsend) {
      PutAlphanumericControl(sendid);
    }
    for (const sendid of arg.boolsend) {
      PutBooleanControl(sendid);
    }
    for (const sendid of arg.labelsend) {
      PutLabelControl(sendid);
    }
    for (const sendid of arg.clicksend) {
      PutClickControl(sendid);
    }
    for (const getid of arg.get) {
      GetControl(getid);
    }
  }
}

function generatePlayButtonCallback(ind) {
  return function() {
    sliderplay[ind].playing = !sliderplay[ind].playing;
    if (sliderplay[ind].playing) {sliderplay[ind].button.innerHTML = "Stop";}
    else {sliderplay[ind].button.innerHTML = "Play";}
  }
}

function PickTimeSeries(seriesID, val) {
  if (seriesID in timeseries) {
    var dat = timeseries[seriesID];
    var times = dat["Times"];
    for (const key in dat["Values"]) {
      var interp = Interpolate(val, times, dat["Values"][key]);
      var valel = document.getElementById(key);
      if (valel != null) {
        valel.value = interp;
      }
      var slider = document.getElementById(key + " Slider");
      if (slider != null) {
        slider.value = interp;
      }
    }
    for (const key in dat["Objects"]) {
      var interp = Interpolate(val, times, dat["Objects"][key]);
      if (key in objects) {
        var updobj = objects[key];
        replaceMatrix(updobj.object,interp);
      }
    }
  }
}

function GetInterp(val, values) {
  var startindex = -1;
  var endindex = -1;
  for (var i=0; i<values.length; i++) {
    var valel = values[i];
    startindex++;
    endindex++;
    if (startindex == 0 && val <= valel) {break;}
    else if (val == valel) {break;}
    else if (val < valel) {startindex = endindex - 1;break;}
  }
  var fractonext = 0;
  if (startindex < endindex) {
    fractonext = (val-values[startindex])/(values[endindex]-values[startindex]);
  }
  return [startindex, fractonext];
}

function Interpolate(val, values, data) {
  const [startindex, fractonext] = GetInterp(val, values);

  if (data[0].constructor === Array) {
    // This is multidimensional
    var matr = Array.from(data[startindex]);
    for (var i=0; i<data[startindex].length; i++) {
      if (fractonext > 0) {matr[i] = data[startindex][i] + fractonext*(data[startindex+1][i]-data[startindex][i]);}
      else {matr[i] = data[startindex][i];}
    }
    return matr
  } else {
    // This is a data range of single values
    var newval = 0;
    if (fractonext > 0) {newval = data[startindex] + fractonext*(data[startindex+1]-data[startindex])}
    else {newval = data[startindex];}
    return newval;
  }
}
