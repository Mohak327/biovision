// Light and dark themes. The choice is stored per browser; without one, the
// system setting decides. index.html applies it before the first paint.
import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
export const THEME_KEY = "biovision.theme";

export function initialTheme(stored: string | null, prefersDark: boolean): Theme {
  if (stored === "light" || stored === "dark") return stored;
  return prefersDark ? "dark" : "light";
}

export const otherTheme = (theme: Theme): Theme => (theme === "dark" ? "light" : "dark");

function readStored(): string | null {
  try {
    return window.localStorage.getItem(THEME_KEY);
  } catch {
    return null; // storage can be blocked
  }
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() =>
    initialTheme(readStored(), window.matchMedia("(prefers-color-scheme: dark)").matches));

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((current) => {
      const next = otherTheme(current);
      try {
        window.localStorage.setItem(THEME_KEY, next);
      } catch {
        // the choice still applies for this visit
      }
      return next;
    });
  }, []);

  return [theme, toggle];
}
