const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const upstream = path.join(root,'integrations/who-is-your-bias-pr19');
const three = path.join(upstream,'node_modules/three');
const { chromium } = require(path.join(upstream,'node_modules/playwright'));
const output = process.argv[2];
if (!output) throw new Error('Usage: capture_avatar_surface.cjs OUTPUT_PREFIX [MODEL.glb]');
const asset = process.argv[3] ? path.resolve(process.argv[3]) : path.join(upstream,'apps/web/public/models/jensen-source-fitted-v3.glb');
const html = `<!doctype html><style>body{margin:0;background:#282b31}</style>
<script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}</script>
<script type="module">
import * as T from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
const size = 800;
const renderer = new T.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
renderer.setSize(size,size);
renderer.setClearColor(0x282b31);
renderer.toneMapping = T.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const scene = new T.Scene();
const ambient = new T.HemisphereLight(0xfff1e7,0xa0a8b7,2.25);
scene.add(ambient);
const key = new T.DirectionalLight(0xffeee0,2.0); key.position.set(-2,4,-4); scene.add(key);
const fill = new T.DirectionalLight(0xd5e1ff,0.85); fill.position.set(3,2,-2); scene.add(fill);
const {scene:model} = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync('/asset.glb');
scene.add(model); scene.updateMatrixWorld(true);
const skin = model.getObjectByName('JensenCandidateSkin');
const camera = new T.PerspectiveCamera(30,1,0.01,10);
const points = skin.geometry.getAttribute('position');
const uv = skin.geometry.getAttribute('uv');
const index = Array.from(skin.geometry.index.array);
const template = (angle=0,glasses=true) => {
  const radians = angle * Math.PI/180;
  camera.position.set(0.68*Math.sin(radians), 1.676, -0.06-0.68*Math.cos(radians));
  camera.lookAt(0,1.657,-0.06); camera.updateMatrixWorld();
  model.traverse(o => { if (o.isMesh && /glasses/i.test(o.name)) o.visible=glasses; });
  renderer.render(scene,camera);
};
template(0,false);
skin.skeleton.update();
const vertex = new T.Vector3();
const screen = Array.from({length:points.count},(_,i) => {
 skin.getVertexPosition(i, vertex).applyMatrix4(skin.matrixWorld).project(camera);
 return [(vertex.x+1)*size/2-0.5,(1-vertex.y)*size/2-0.5,vertex.z, uv.getX(i),uv.getY(i)];
});
window.capture=template; window.projection={screen,index,size};
</script>`;
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath: process.env.YAICON_CHROMIUM,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try {
  const page=await browser.newPage({viewport:{width:800,height:800}});
  page.on('pageerror',e=>console.error(e));
  await page.route('http://avatar.local/**',async route=>{
   const key = new URL(route.request().url()).pathname;
   if(key==='/') return route.fulfill({contentType:'text/html',body:html});
   if(key==='/asset.glb') return route.fulfill({path:asset,contentType:'model/gltf-binary'});
   if(key.startsWith('/three/')) return route.fulfill({path:path.join(three,key.slice(7)),contentType:'text/javascript'});
   await route.abort();
  });
  await page.goto('http://avatar.local/');
  await page.waitForFunction(()=>window.projection);
  fs.writeFileSync(output+'-geometry.json',JSON.stringify(await page.evaluate(()=>window.projection)));
  await page.screenshot({path:output+'-neutral.png'});
  await page.evaluate(()=>window.capture(0,true));
  await page.screenshot({path:output+'-front.png'});
  await page.evaluate(()=>window.capture(25,true));
  await page.screenshot({path:output+'-angle.png'});
 } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
