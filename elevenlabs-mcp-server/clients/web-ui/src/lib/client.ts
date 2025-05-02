import { ElevenLabsClient } from "./elevenlabs-client";
import { browser } from "$app/environment";
import { env } from "$env/dynamic/private";

// Only initialize the client on the server side
export const elevenlabsClient = !browser
  ? new ElevenLabsClient("uv", [
      "--directory",
      env.MCP_SERVER_DIR || "",
      "run",
      "elevenlabs-mcp",
    ])
  : null;

// Export the client type for use in components
export type { JobHistory } from "./elevenlabs-client";
