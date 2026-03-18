import "../src/stories/storybook.css"

const preview = {
  parameters: {
    layout: "fullscreen",
    backgrounds: {
      default: "control-room",
      values: [
        { name: "control-room", value: "#0b1220" },
        { name: "panel", value: "#1f2937" },
      ],
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
      },
    },
  },
}

export default preview
