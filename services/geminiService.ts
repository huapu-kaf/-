import { GoogleGenAI, Modality } from "@google/genai";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

// Refined prompts for better data augmentation consistency.
const AUGMENTATION_PROMPTS = [
  "from a slightly lower angle.",
  "from a slightly higher angle.",
  "a close-up shot, focusing on a key detail.",
  "from the left side.",
  "from the right side.",
  "with brighter, more even lighting.",
  "with dramatic, high-contrast lighting.",
  "with a warmer color tone.",
  "with a cooler color tone.",
  "with a shallow depth of field, blurring the background.",
];


export const editImageWithPrompt = async (
  base64ImageData: string,
  mimeType: string,
  prompt: string
): Promise<string> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: [
          {
            inlineData: {
              data: base64ImageData,
              mimeType: mimeType,
            },
          },
          {
            text: prompt,
          },
        ],
      },
      config: {
          responseModalities: [Modality.IMAGE],
      },
    });

    if (response.promptFeedback?.blockReason) {
      throw new Error(
        `Request was blocked with reason: ${response.promptFeedback.blockReason}. ${response.promptFeedback.blockReasonMessage || ''}`
      );
    }

    if (response.candidates && response.candidates.length > 0) {
      for (const candidate of response.candidates) {
        for (const part of candidate.content.parts) {
          if (part.inlineData) {
            return part.inlineData.data;
          }
        }
      }
    }

    if (response.text) {
      throw new Error(`Model returned a text response instead of an image: "${response.text}"`);
    }

    throw new Error("No image data found in the Gemini API response.");

  } catch (error) {
    console.error("Error calling Gemini API:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to generate image: ${errorMessage}`);
  }
};

export const generateImageDataset = async (
  base64ImageData: string,
  mimeType: string,
): Promise<string[]> => {
  // A more explicit prompt to maintain subject consistency.
  const basePrompt = "Keeping the main subject from the original photo identical, generate a new photorealistic image of it, but change the perspective to be ";

  const generationPromises = AUGMENTATION_PROMPTS.map(perspective => {
    const fullPrompt = basePrompt + perspective;
    return editImageWithPrompt(base64ImageData, mimeType, fullPrompt);
  });

  try {
    const results = await Promise.all(generationPromises);
    return results;
  } catch (error) {
    console.error("Error generating image dataset:", error);
    throw new Error("Failed to generate the full image dataset. One or more images could not be created. Please try again.");
  }
};