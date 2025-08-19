/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        quantys: {
          blue: '#1E2A78',       // Bleu Profond : Fond / Sidebar
          light: '#4F7CAC',      // Bleu Clair : Hover, icons, borders
          gray: '#E5E8EB',       // Gris Neutre : Fond clair, séparateurs
          yellow: '#F4D35E',     // Jaune : KPI, alertes, actions
          white: '#FFFFFF',      // Blanc : Fond, texte clair
        },
      },
      fontFamily: {
        sans: ['Poppins', 'sans-serif'],
      },
      keyframes: {
        // Définition de l'animation 'fadeIn'
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        // Définition de l'animation 'scaleIn' pour le pop-up
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
      animation: {
        // Association des keyframes à des noms de classes utilitaires
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'scale-in': 'scaleIn 0.3s ease-out forwards', // Animation plus rapide pour le pop-up
      },
    },
  },
  plugins: [],
}
