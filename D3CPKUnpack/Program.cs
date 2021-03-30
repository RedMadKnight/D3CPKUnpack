using System;
using System.Collections.Generic;
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
        public static Dictionary<uint, cpk.CompressedSectorChunk> DictCompressedSectorChunk;
        public static cpk.CompressedSectorChunk[] CompressedSectorChunk;
        public static int rev;


        public static void WriteHeader()
        {
            Console.WriteLine("Header");
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
            Console.WriteLine("Total sectors :\t" + HeaderStruct.CompSectorCount.ToString());
            Console.WriteLine("First Sector Position :\t" + HeaderStruct.FirstSectorPosition.ToString("X8"));
        }

        public static void WriteFileInfo(int i)
        {
            Console.WriteLine("FileInfo");
            Console.WriteLine(i.ToString("d5") + ": dwHash :\t" + SortedFileInfo[i].dwHash.ToString("X16"));
            Console.WriteLine(i.ToString("d5") + ": nSize :\t" + SortedFileInfo[i].nSize.ToString());
            Console.WriteLine(i.ToString("d5") + ": nLocationCount :\t" + SortedFileInfo[i].nLocationCount.ToString());
            Console.WriteLine(i.ToString("d5") + ": nLocationIndex :\t" + SortedFileInfo[i].nLocationIndex.ToString());
        }

        public static void WriteLocations(int i)
        {
            Console.WriteLine("Location");
            Console.WriteLine(i.ToString("d3") + ": index :\t" + Locations[i].index.ToString("d5"));
            Console.WriteLine(i.ToString("d3") + ": offset :\t" + Locations[i].offset.ToString("X16"));
        }

        public static uint FindLocationIndex(uint i)
        {
            uint a;
            for (a = 0; a < Locations.Length; a++)
                if (Locations[a].index == i)
                    break;
            return a;
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
            Console.WriteLine("FileName");
            Console.WriteLine(i.ToString("d3") + ": offset :\t" + FileName[i].offset.ToString("X8"));
            Console.WriteLine(i.ToString("d3") + ": fileHash :\t" + FileName[i].fileHash.ToString("X16"));
            Console.WriteLine(i.ToString("d3") + ": filename :\t" + FileName[i].filename.ToString());
        }

        public static void WriteChunckSectorInfo(int i)
        {
            Console.WriteLine("ChunckSector");
            Console.WriteLine(i.ToString("d3") + ": nr :\t" + CompressedSectorChunk[i].nr.ToString("d6"));
            Console.WriteLine(i.ToString("d3") + ": OffsetPosition :\t" + CompressedSectorChunk[i].position.ToString("X8"));
            Console.WriteLine(i.ToString("d3") + ": CompChunkSize :\t" + CompressedSectorChunk[i].CompChunkSize.ToString("d6"));
            Console.WriteLine(i.ToString("d3") + ": DecompChunkSize :\t" + CompressedSectorChunk[i].DecompChunkSize.ToString("d6"));
            Console.WriteLine(i.ToString("d3") + ": flag :\t" + CompressedSectorChunk[i].flag.ToString("d6"));
            Console.WriteLine(i.ToString("d3") + ": Sector :\t" + CompressedSectorChunk[i].CompSector.ToString("d6"));
            Console.WriteLine(i.ToString("d3") + ": DecompStartOffset :\t" + CompressedSectorChunk[i].StartDecompOffset.ToString("X10"));
        }

        public static byte[] GetChunck(FileStream s, int i, int rev)
        {
            helper help = new helper();
            return help.DecompressChunk(s, (int)CompressedSectorChunk[i].position, rev);
        }


        static void Main(string[] args)
        {
            string path = "";
            if (args.Length == 0)
                path = "D:\\Common.cpk";
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
            DictCompressedSectorChunk = cpk.CompressedSectorChunk.ReadSectors(fs);
            CompressedSectorChunk = cpk.CompressedSectorChunk.Read_CompressedSectorChunk(DictCompressedSectorChunk);

            WriteHeader();
            WriteLocations(1171);
            WriteFileInfo(1171);
            WriteFileName(1171);
            for (uint idx = 2; idx < 78; idx++)
                if (CompressedSectorChunk[idx].flag != 3)
                {
                    Console.WriteLine(idx.ToString("d3"));
                    break;
                }
            fs.Close();
            Console.ReadKey();
        }
    }
}