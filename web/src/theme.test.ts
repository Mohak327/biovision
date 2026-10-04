import { describe, expect, it } from "vitest";
import { initialTheme, otherTheme } from "./theme";

describe("theme", () => {
  it("uses a stored choice over the system setting", () => {
    expect(initialTheme("light", true)).toBe("light");
    expect(initialTheme("dark", false)).toBe("dark");
  });

  it("follows the system setting when nothing valid is stored", () => {
    expect(initialTheme(null, true)).toBe("dark");
    expect(initialTheme(null, false)).toBe("light");
    expect(initialTheme("purple", true)).toBe("dark");
  });

  it("switches between the two themes", () => {
    expect(otherTheme("light")).toBe("dark");
    expect(otherTheme("dark")).toBe("light");
  });
});
