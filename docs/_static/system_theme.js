if (localStorage.getItem("theme") === null) {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    localStorage.setItem("theme", prefersDark ? "dark" : "light");
}
