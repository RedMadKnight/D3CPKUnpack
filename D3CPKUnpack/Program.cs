using System;
using System.IO;

namespace D3CPKUnpack
{
    class Program
    {
        public static cpk.HeaderStruct HeaderStruct;
        public static cpk.SortedFileInfo[] SortedFileInfo;
        public static cpk.Locations[] Locations;
        public static cpk.CompressedSectorToDecompressedOffset[] CompressedSectorToDecompressedOffset;
        public static cpk.DecompressedSectorToCompressedSector[] DecompressedSectorToCompressedSector;
        public static cpk.FileName[] FileName;
        public static int rev;


        public static void WriteHeader()
        {
            Console.WriteLine("MagicNumber :\t" + HeaderStruct.MagicNumber.ToString("X8"));
            Console.WriteLine("PackageVersion :\t" + HeaderStruct.PackageVersion.ToString());
            Console.WriteLine("DecompressedFileSize :\t" + HeaderStruct.DecompressedFileSize.ToString());
            Console.WriteLine("Flags :\t" + HeaderStruct.Flags.ToString("d1"));
            Console.WriteLine("FileCount :\t" + HeaderStruct.FileCount.ToString());
            Console.WriteLine("LocationCount :\t" + HeaderStruct.LocationCount.ToString());
            Console.WriteLine("HeaderSector :\t" + HeaderStruct.HeaderSector.ToString());
            Console.WriteLine("FileSizeBitCount :\t" + HeaderStruct.FileSizeBitCount.ToString());
            Console.WriteLine("FileLocationCountBitCount :\t" + HeaderStruct.FileLocationCountBitCount.ToString());
            Console.WriteLine("FileLocationIndexBitCount :\t" + HeaderStruct.FileLocationIndexBitCount.ToString());
            Console.WriteLine("LocationBitCount :\t" + HeaderStruct.LocationBitCount.ToString());
            Console.WriteLine("CompSectorToDecomOffsetBitCount :\t" + HeaderStruct.CompSectorToDecomOffsetBitCount.ToString());
            Console.WriteLine("DecompSectorToCompSectorBitCount :\t" + HeaderStruct.DecompSectorToCompSectorBitCount.ToString());
            Console.WriteLine("CRC :\t" + HeaderStruct.CRC.ToString());
            Console.WriteLine("ReadSectorSize :\t0x" + HeaderStruct.ReadSectorSize.ToString("X4"));
            Console.WriteLine("CompSectorSize :\t0x" + HeaderStruct.CompSectorSize.ToString("X4"));
        }

        public static void WriteFileInfo(int i)
        {
            Console.WriteLine(i.ToString("d5") + ": dwHash :\t" + SortedFileInfo[i].dwHash.ToString("X16"));
            Console.WriteLine(i.ToString("d5") + ": nSize :\t" + SortedFileInfo[i].nSize.ToString());
            Console.WriteLine(i.ToString("d5") + ": nLocationCount :\t" + SortedFileInfo[i].nLocationCount.ToString());
            Console.WriteLine(i.ToString("d5") + ": nLocationIndex :\t" + SortedFileInfo[i].nLocationIndex.ToString());
            Console.ReadKey();
        }

        public static void WriteLocations(int i)
        {
            Console.WriteLine(i.ToString("d3") + ": index :\t" + Locations[i].index.ToString("d5"));
            Console.WriteLine(i.ToString("d3") + ": offset :\t" + Locations[i].offset.ToString("X16"));
        }

        public static void WriteCompressedSectorToDecompressedOffset(int i)
        {
            Console.WriteLine(i.ToString("d3") + ": SectorIndex :\t" + CompressedSectorToDecompressedOffset[i].SectorIndex.ToString("d5"));
            Console.WriteLine(i.ToString("d3") + ": DecompressedOffset :\t" + CompressedSectorToDecompressedOffset[i].DecompressedOffset.ToString("X8"));
        }

        public static void WriteDecompressedSectorToCompressedSector(int i)
        {
            Console.WriteLine(i.ToString("d3") + ": DecompressedSector :\t" + DecompressedSectorToCompressedSector[i].DecompressedSector.ToString("d5"));
            Console.WriteLine(i.ToString("d3") + ": CompressedSector :\t" + DecompressedSectorToCompressedSector[i].CompressedSector.ToString("d5"));
        }

        public static void WriteFileName(int i)
        {
            Console.WriteLine(i.ToString("d3") + ": offset :\t" + FileName[i].offset.ToString("X8"));
            Console.WriteLine(i.ToString("d3") + ": fileHash :\t" + FileName[i].fileHash.ToString("X16"));
            Console.WriteLine(i.ToString("d3") + ": filename :\t" + FileName[i].filename.ToString());
        }

        static void Main(string[] args)
        {
            string path = "";
            if (args.Length == 0)
                 //path = "C:\\ServerCommon.cpk";
            path = "C:\\plPL_CacheCommon.cpk";
            else
                path = args[0];
            helper help = new helper();
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            fs.Seek(0, 0);
            rev = cpk.IsByteReverse(fs);
            HeaderStruct = cpk.HeaderStruct.ReadHeader(fs, rev);
            SortedFileInfo = cpk.SortedFileInfo.Read_SortedFileInfo(fs);
            Locations = cpk.Locations.Read_Locations(fs);
            CompressedSectorToDecompressedOffset = cpk.CompressedSectorToDecompressedOffset.Read_CompressedSectorToDecompressedOffset(fs);
            DecompressedSectorToCompressedSector = cpk.DecompressedSectorToCompressedSector.Read_DecompressedSectorToCompressedSector(fs);
            FileName = cpk.FileName.Read_FileName(fs);
            WriteHeader();        
            //WriteLocations(10);
            //WriteCompressedSectorToDecompressedOffset(CompressedSectorToDecompressedOffset.Length-1);
            //WriteDecompressedSectorToCompressedSector(10);
            //WriteFileName(0);
            fs.Close();
            Console.ReadKey();
        }
    }
}