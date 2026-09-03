import { FileBlob } from "@oai/artifact-tool";
const b = await FileBlob.load("/Users/corentinplumet/Documents/RL_Marl2grid/latex/figures/sparse_control_fulltest_tradeoff.png");
console.log(b);
console.log(Object.getOwnPropertyNames(b));
console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(b)));
