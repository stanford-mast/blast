/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'blast-yellow': '#FFE067',
        'dark': '#1F1F1F',
      },
    },
  },
  plugins: [],
}