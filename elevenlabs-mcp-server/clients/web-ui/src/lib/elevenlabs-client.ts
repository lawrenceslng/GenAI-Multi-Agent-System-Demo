import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  CallToolResultSchema,
  ReadResourceResultSchema,
} from "@modelcontextprotocol/sdk/types.js";
import type {
  TextContent,
  EmbeddedResource,
  CallToolRequest,
  CallToolResult,
  BlobResourceContents,
  TextResourceContents,
  ReadResourceRequest,
  ReadResourceResult,
} from "@modelcontextprotocol/sdk/types.js";

export interface JobHistory {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  script_parts: ScriptPart[];
  output_file?: string;
  error?: string;
  created_at: string;
  updated_at: string;
  total_parts: number;
  completed_parts: number;
}

export interface Voice {
  voice_id: string;
  name: string;
  category?: string;
  description?: string;
  labels?: Record<string, string>;
  preview_url?: string;
  is_default?: boolean;
}

export interface AudioGenerationResponse {
  success: boolean;
  message: string;
  debugInfo: string[];
  audioData?: {
    uri: string;
    name: string;
    data: string; // base64 encoded audio
  };
}

interface ScriptInterface {
  script: ScriptPart[];
}

export interface ScriptPart {
  text: string;
  voice_id?: string;
  actor?: string;
}

export class ElevenLabsClient {
  private client: Client;
  private connectionPromise: Promise<void>;

  constructor(command: string, args?: string[]) {
    const transport = new StdioClientTransport({
      command,
      args,
    });

    console.log("command:", command, "args:", args);

    this.client = new Client(
      {
        name: "elevenlabs-client",
        version: "0.1.0",
      },
      {
        capabilities: {},
      }
    );

    // Initialize connection promise
    this.connectionPromise = this.client.connect(transport);
    this.connectionPromise.catch((error: Error) => {
      console.error("Failed to connect:", error);
    });
  }

  private parseHistoryResponse(response: ReadResourceResult): JobHistory[] {
    console.log("Raw history response:", response);

    for (const content of response.contents) {
      if (
        content.mimeType === "text/plain" &&
        typeof content.text === "string"
      ) {
        try {
          // Clean up the text content by removing any leading/trailing whitespace and quotes
          const cleanText = content.text.trim().replace(/^['"]|['"]$/g, "");
          const parsed = JSON.parse(cleanText);
          if (Array.isArray(parsed)) {
            return parsed as JobHistory[];
          } else if (typeof parsed === "object" && parsed !== null) {
            // If we got a single job object, wrap it in an array
            return [parsed as JobHistory];
          }
        } catch (error) {
          console.error("Error parsing history response:", error);
          // Try to parse as a string concatenation
          try {
            const concatenatedJson = content.text
              .split("\n")
              .map((line) => line.trim())
              .join("")
              .replace(/^['"]|['"]$/g, "")
              .replace(/\s*\+\s*/g, "");
            console.log(
              "Attempting to parse concatenated JSON:",
              concatenatedJson
            );
            const parsed = JSON.parse(concatenatedJson);
            if (Array.isArray(parsed)) {
              return parsed as JobHistory[];
            } else if (typeof parsed === "object" && parsed !== null) {
              return [parsed as JobHistory];
            }
          } catch (innerError) {
            console.error("Error parsing concatenated JSON:", innerError);
          }
        }
      }
    }
    return [];
  }

  private parseToolResponse(response: CallToolResult): AudioGenerationResponse {
    const result: AudioGenerationResponse = {
      success: false,
      message: "",
      debugInfo: [],
    };

    if (response.content) {
      for (const content of response.content) {
        if (content.type === "text") {
          // Split the text content into lines
          const lines = content.text.split("\n");
          // First line is the status message
          result.message = lines[0];
          // Rest are debug info (skip the "Debug info:" line)
          result.debugInfo = lines.slice(2);
          // Check if the message indicates success
          result.success = result.message.includes("successful");
        } else if (content.type === "resource") {
          const resource = content.resource as BlobResourceContents;
          result.audioData = {
            uri: resource.uri,
            name:
              (resource.name as string) ||
              resource.uri.split("/").pop() ||
              "audio",
            data: resource.blob,
          };
        }
      }
    }

    return result;
  }

  async generateSimpleAudio(
    text: string,
    voice_id?: string
  ): Promise<AudioGenerationResponse> {
    try {
      // Wait for connection before making request
      await this.connectionPromise;

      const request: CallToolRequest = {
        method: "tools/call",
        params: {
          name: "generate_audio_simple",
          arguments: {
            text,
            voice_id,
          },
        },
      };

      const response = await this.client.request(request, CallToolResultSchema);
      return this.parseToolResponse(response);
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        message: `Error generating audio: ${errorMessage}`,
        debugInfo: [],
      };
    }
  }

  async generateScriptAudio(
    script: string | ScriptInterface
  ): Promise<AudioGenerationResponse> {
    try {
      // Wait for connection before making request
      await this.connectionPromise;

      const scriptJson =
        typeof script === "string" ? script : JSON.stringify(script);

      const request: CallToolRequest = {
        method: "tools/call",
        params: {
          name: "generate_audio_script",
          arguments: {
            script: scriptJson,
          },
        },
      };

      const response = await this.client.request(request, CallToolResultSchema);
      return this.parseToolResponse(response);
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        message: `Error generating audio: ${errorMessage}`,
        debugInfo: [],
      };
    }
  }

  async getJobHistory(): Promise<JobHistory[]> {
    try {
      await this.connectionPromise;

      const request: ReadResourceRequest = {
        method: "resources/read",
        params: {
          uri: "voiceover://history",
        },
      };

      const response = await this.client.request(
        request,
        ReadResourceResultSchema
      );
      return this.parseHistoryResponse(response);
    } catch (error) {
      console.error("Error fetching job history:", error);
      return [];
    }
  }

  async getJobById(jobId: string): Promise<JobHistory | null> {
    try {
      await this.connectionPromise;

      const request: ReadResourceRequest = {
        method: "resources/read",
        params: {
          uri: `voiceover://history/${jobId}`,
        },
      };

      const response = await this.client.request(
        request,
        ReadResourceResultSchema
      );
      const jobs = this.parseHistoryResponse(response);
      return jobs.length > 0 ? jobs[0] : null;
    } catch (error) {
      console.error("Error fetching job:", error);
      return null;
    }
  }

  async getAudioFile(jobId: string): Promise<{
    success: boolean;
    audioData?: {
      data: string;
      name: string;
      mimeType: string;
    };
    error?: string;
  }> {
    try {
      await this.connectionPromise;

      const request: CallToolRequest = {
        method: "tools/call",
        params: {
          name: "get_audio_file",
          arguments: {
            job_id: jobId,
          },
        },
      };

      const response = await this.client.request(request, CallToolResultSchema);

      // Check for error message
      for (const content of response.content) {
        if (content.type === "text") {
          return {
            success: false,
            error: content.text,
          };
        }
      }

      // Look for audio file content
      for (const content of response.content) {
        if (content.type === "resource") {
          const resource = content.resource as BlobResourceContents;
          return {
            success: true,
            audioData: {
              data: resource.blob,
              name: resource.name as string,
              mimeType: resource.mimeType || "audio/mpeg",
            },
          };
        }
      }

      return {
        success: false,
        error: "No audio content found in response",
      };
    } catch (error) {
      console.error("Error getting audio file:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async deleteJob(jobId: string): Promise<boolean> {
    try {
      await this.connectionPromise;

      const request: CallToolRequest = {
        method: "tools/call",
        params: {
          name: "delete_job",
          arguments: {
            job_id: jobId,
          },
        },
      };

      const response = await this.client.request(request, CallToolResultSchema);

      // Check if deletion was successful based on response message
      for (const content of response.content) {
        if (content.type === "text") {
          return content.text.includes("Successfully deleted");
        }
      }
      return false;
    } catch (error) {
      console.error("Error deleting job:", error);
      return false;
    }
  }

  async getVoices(): Promise<Voice[]> {
    try {
      await this.connectionPromise;

      const request: ReadResourceRequest = {
        method: "resources/read",
        params: {
          uri: "voiceover://voices",
        },
      };

      const response = await this.client.request(request, ReadResourceResultSchema);
      
      for (const content of response.contents) {
        if (content.mimeType === "text/plain" && typeof content.text === "string") {
          try {
            const voices = JSON.parse(content.text);
            if (Array.isArray(voices)) {
              return voices as Voice[];
            }
          } catch (error) {
            console.error("Error parsing voices response:", error);
          }
        }
      }
      return [];
    } catch (error) {
      console.error("Error fetching voices:", error);
      return [];
    }
  }

  async close(): Promise<void> {
    try {
      // Wait for connection before closing
      await this.connectionPromise;
      if (this.client) {
        await this.client.close();
      }
    } catch (error) {
      console.error("Error closing client:", error);
    }
  }
}
