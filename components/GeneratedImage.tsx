
import React from 'react';
import DownloadIcon from './icons/DownloadIcon';

interface GeneratedImageProps {
  imageSrc: string;
  index: number;
  originalMimeType: string;
}

const GeneratedImage: React.FC<GeneratedImageProps> = ({ imageSrc, index, originalMimeType }) => {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageSrc;
    const fileExtension = originalMimeType.split('/')[1] || 'png';
    link.download = `generated_image_${index + 1}.${fileExtension}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="relative group aspect-square rounded-lg overflow-hidden shadow-lg border-2 border-gray-700/50">
      <img src={imageSrc} alt={`Generated Image ${index + 1}`} className="w-full h-full object-cover" />
      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-60 transition-all duration-300 flex items-center justify-center">
        <button
          onClick={handleDownload}
          className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 text-gray-900 font-bold py-2 px-4 rounded-full flex items-center gap-2 transform hover:scale-110 ease-in-out"
          aria-label={`Download image ${index + 1}`}
        >
          <DownloadIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default GeneratedImage;
