import { PresentationFile, FileBlob } from "@oai/artifact-tool";

const file = "/Users/corentinplumet/Documents/RL_Marl2grid/tmp/thesis_defense_build/epfl_inspect/template-starter.pptx";
const p = await PresentationFile.importPptx(await FileBlob.load(file));
const slide = p.slides.items[6];
console.log(JSON.stringify({
  slides: p.slides.items.length,
  shapes: slide.shapes.items.map((x) => ({name: x.name, text: x.text, frame: x.frame})),
  images: slide.images.items.map((x) => ({name: x.name, frame: x.frame})),
  imageDetails: slide.images.items.map((x) => ({name: x.name, crop: x.crop, fit: x.fit, geometry: x.geometry})),
}, null, 2));
