import { PresentationFile, FileBlob } from "@oai/artifact-tool";
const p = await PresentationFile.importPptx(await FileBlob.load("/Users/corentinplumet/Documents/RL_Marl2grid/tmp/thesis_defense_build/epfl_inspect/template-starter.pptx"));
for (const i of [1, 6]) {
  const s = p.slides.items[i];
  console.log(i + 1, Object.getOwnPropertyNames(s), s.layout, s.layoutId);
}
console.log("layouts", p.layouts.items.length, p.layouts.items.slice(0,5).map(x=>({id:x.id,name:x.name})));
