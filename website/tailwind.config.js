/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // The design tokens in styles/globals.css.
        page: '#121212',
        surface: '#1c1c1d',
        'surface-hover': '#262626',
        border: '#2a2a2a',
        ink: '#fafafa',
        'ink-secondary': '#c8c8c8',
        'ink-tertiary': '#808080',
        brand: '#f2ff2d',
      },
      fontFamily: {
        display: [
          'var(--font-zalando-semi-expanded)',
          'var(--font-inter)',
          'sans-serif',
        ],
        sans: ['var(--font-inter)', 'sans-serif'],
        mono: ['var(--font-sometype-mono)', 'monospace'],
      },
      // The whole radius scale is zero rather than special-casing individual
      // components, so nothing accidentally picks up a rounded-* utility.
      borderRadius: {
        none: '0px',
        sm: '0px',
        DEFAULT: '0px',
        md: '0px',
        lg: '0px',
        xl: '0px',
        '2xl': '0px',
        '3xl': '0px',
        full: '0px',
      },
      transitionDuration: {
        // The hover cadence every link and button uses (`duration-[50ms]`).
        50: '50ms',
      },
    },
  },
  plugins: [],
}
