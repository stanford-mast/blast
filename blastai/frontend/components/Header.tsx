import Image from 'next/image';

export const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 bg-[#1f1f1f] z-50 px-4 py-4 flex items-center">
      <div className="flex items-center gap-3">
        <Image 
          src="/assets/blast_icon_only.svg" 
          alt="BLAST Logo" 
          width={32} 
          height={32}
          className="w-8 h-8"
        />
        <span className="text-xl font-semibold font-['Inter']">BLAST</span>
      </div>
    </header>
  );
};