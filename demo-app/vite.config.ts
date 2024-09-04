import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import * as packageJson from "../package.json";

export default defineConfig({
    plugins: [react()],
    assetsInclude: ["**/*.md"],
    base: "/",
    build: {
        outDir: "dist",
        emptyOutDir: true,
    },
    define: {
        "import.meta.env.PACKAGE_VERSION": JSON.stringify(packageJson.version),
    },
});
