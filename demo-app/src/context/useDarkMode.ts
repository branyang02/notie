import { useContext } from "react"
import { DarkModeContext } from "./DarkModeContext"

export const useDarkMode = () => {
    const context = useContext(DarkModeContext)
    if (context === undefined) {
        throw new Error("useDarkMode must be used within a DarkModeProvider")
    }
    return context
}
