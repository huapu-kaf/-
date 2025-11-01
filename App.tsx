
import React, { useState, useCallback } from 'react';
import { generateImageDataset } from './services/geminiService';
import { fileToBase64 } from './utils/fileUtils';
import type { UploadedImage } from './types';
import ImageDisplay from './components/ImageDisplay';
import GeneratedImage from './components/GeneratedImage';
import DownloadIcon from './components/icons/DownloadIcon';

const App: React.FC = () => {
  const [originalImage, setOriginalImage] = useState<UploadedImage | null>(null);
  const [editedImages, setEditedImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      try {
        setError(null);
        setEditedImages([]); // Clear previous dataset
        const base64 = await fileToBase64(file);
        setOriginalImage({
          file: file,
          base64: base64,
          mimeType: file.type,
        });
      } catch (err) {
        setError('Failed to load image. Please try again.');
        setOriginalImage(null);
      }
    } else {
      setError('Please select a valid image file.');
      setOriginalImage(null);
    }
  };

  const handleGenerate = useCallback(async () => {
    if (!originalImage) {
      setError('Please upload an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setEditedImages([]);

    try {
      const newImageBase64Array = await generateImageDataset(
        originalImage.base64,
        originalImage.mimeType
      );
      const imageDataUrls = newImageBase64Array.map(
        (base64) => `data:${originalImage.mimeType};base64,${base64}`
      );
      setEditedImages(imageDataUrls);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during image generation.');
    } finally {
      setIsLoading(false);
    }
  }, [originalImage]);

  const handleDownloadAll = () => {
    if (!originalImage) return;
    editedImages.forEach((src, index) => {
        const link = document.createElement('a');
        link.href = src;
        const fileExtension = originalImage.mimeType.split('/')[1] || 'png';
        link.download = `generated_image_${index + 1}.${fileExtension}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col items-center p-4 sm:p-6 md:p-8">
      <div className="w-full max-w-7xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-600">
            Gemini Dataset Generator
          </h1>
          <p className="mt-2 text-lg text-gray-400">
            Upload an image to generate 10 variations with different perspectives for your dataset.
          </p>
        </header>

        <main className="w-full">
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 shadow-2xl border border-gray-700">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
              <div>
                <label htmlFor="file-upload" className="block text-sm font-medium text-gray-300 mb-2">1. Upload Your Image</label>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-700 cursor-pointer"
                />
              </div>
              <div className="flex items-center justify-center">
                <button
                  onClick={handleGenerate}
                  disabled={isLoading || !originalImage}
                  className="w-full md:w-auto text-lg font-bold py-4 px-8 rounded-full bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg hover:from-teal-600 hover:to-blue-600 transition-all duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 disabled:scale-100 flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Generating Dataset...
                    </>
                  ) : '✨ Generate 10 Images'}
                </button>
              </div>
            </div>
          </div>
          
          {error && <div className="mt-6 bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-center">{error}</div>}

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
            <ImageDisplay title="Original Image" imageSrc={originalImage ? `data:${originalImage.mimeType};base64,${originalImage.base64}` : null} />
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-semibold text-gray-300 text-center">Generated Dataset</h2>
               <div className="w-full min-h-[calc(100%-2rem)] aspect-square bg-gray-800 rounded-xl border-2 border-dashed border-gray-600 flex items-center justify-center overflow-hidden relative shadow-inner p-4">
                {isLoading && (
                  <div className="absolute inset-0 bg-gray-900/70 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                    <svg className="animate-spin h-10 w-10 text-sky-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="mt-4 text-lg text-gray-300">Generating 10 images...</p>
                  </div>
                )}
                {!isLoading && editedImages.length > 0 && (
                  <div className='w-full h-full flex flex-col gap-4'>
                    <button
                      onClick={handleDownloadAll}
                      className="w-full text-md font-bold py-2 px-6 rounded-lg bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-md hover:from-teal-600 hover:to-blue-600 transition-all duration-300 ease-in-out flex items-center justify-center gap-2"
                    >
                      <DownloadIcon className="w-5 h-5"/>
                      Download All
                    </button>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 flex-grow">
                      {editedImages.map((src, index) => (
                        <GeneratedImage key={index} imageSrc={src} index={index} originalMimeType={originalImage!.mimeType} />
                      ))}
                    </div>
                  </div>
                )}
                {!isLoading && editedImages.length === 0 && (
                  <div className="text-center text-gray-500">
                    <p className="mt-2 text-sm">Generated images will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
