import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import dts from "vite-plugin-dts";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";
import wasmPack from "vite-plugin-wasm-pack";

export default defineConfig({
    build: {
        lib: {
            entry: path.resolve(__dirname, "src/index.ts"),
            name: "notie",
            fileName: (format) => `index.${format}.js`,
        },
        rollupOptions: {
            external: ["react", "react-dom"],
            output: {
                globals: {
                    react: "React",
                    "react-dom": "ReactDOM",
                },
            },
        },
        sourcemap: true,
        emptyOutDir: true,
    },
    plugins: [react(), dts(), cssInjectedByJsPlugin(), wasmPack("./markdown_processor")],
    assetsInclude: ["**/*.md"],
});
