const react = require("react/package.json");
const reactDom = require("react-dom/package.json");

const reactMajor = react.version.split(".")[0];
const reactDomMajor = reactDom.version.split(".")[0];

if (reactMajor !== reactDomMajor) {
  console.error(
    `Version mismatch: react=${react.version}, react-dom=${reactDom.version}`
  );
  process.exit(1);
}

console.log(`build ok with react ${react.version} and react-dom ${reactDom.version}`);
