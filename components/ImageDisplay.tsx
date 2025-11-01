
import React from 'react';
import ImageIcon from './icons/ImageIcon';

interface ImageDisplayProps {
  title: string;
  imageSrc: string | null;
  isLoading?: boolean;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ title, imageSrc, isLoading = false }) => {
  return (
    <div className="flex flex-col items-center gap-4">
      <h2 className="text-xl font-semibold text-gray-300">{title}</h2>
      <div className="w-full aspect-square bg-gray-800 rounded-xl border-2 border-dashed border-gray-600 flex items-center justify-center overflow-hidden relative shadow-inner">
        {isLoading && (
          <div className="absolute inset-0 bg-gray-900/70 backdrop-blur-sm flex flex-col items-center justify-center z-10">
            <svg className="animate-spin h-10 w-10 text-pink-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="mt-4 text-lg text-gray-300">Generating...</p>
          </div>
        )}
        {imageSrc ? (
          <img src={imageSrc} alt={title} className="w-full h-full object-contain" />
        ) : !isLoading && (
          <div className="text-center text-gray-500">
            <ImageIcon className="mx-auto h-16 w-16" />
            <p className="mt-2 text-sm">Image will appear here</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageDisplay;
