import { slides_v1 } from 'googleapis';
import { CreatePresentationArgs } from '../schemas.js';
import { handleGoogleApiError } from '../utils/errorHandler.js';

/**
 * Creates a new Google Slides presentation.
 * @param slides The authenticated Google Slides API client.
 * @param args The arguments for creating the presentation.
 * @returns A promise resolving to the MCP response.
 * @throws McpError if the Google API call fails.
 */
export const createPresentationTool = async (slides: slides_v1.Slides, args: CreatePresentationArgs) => {
  try {
    const response = await slides.presentations.create({
      requestBody: {
        title: args.title,
      },
    });
    // Return a simpler JSON object containing only the presentationId
    const result = {
      presentationId: response.data.presentationId,
      title: response.data.title,
    };
    return {
      // Return the simplified result as a JSON string in the text content
      content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
    };
  } catch (error: unknown) {
    throw handleGoogleApiError(error, 'create_presentation');
  }
};
