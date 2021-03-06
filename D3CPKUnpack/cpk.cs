﻿using System;
using System.Collections.Generic;
using System.IO;

namespace D3CPKUnpack
{
    class cpk
    {
        public class HeaderStruct
        {
            public uint MagicNumber;                           // 4 - bits ->  0 -  3
            public uint PackageVersion;                        // 4 - bits ->  4 -  7
            public ulong DecompressedFileSize;                 // 8 - bits ->  8 - 15
            public uint Flags;                                 // 4 - bits -> 16 - 19 | P1
            public uint FileCount;                             // 4 - bits -> 20 - 23 | P1
            public uint LocationCount;                         // 4 - bits -> 24 - 27
            public uint HeaderSector;                          // 4 - bits -> 28 - 31
            public uint FileSizeBitCount;                      // 4 - bits -> 32 - 35 | P2
            public uint FileLocationCountBitCount;             // 4 - bits -> 36 - 39 | P2
            public uint FileLocationIndexBitCount;             // 4 - bits -> 40 - 43 | P3
            public uint LocationBitCount;                      // 4 - bits -> 44 - 47 | P3
            public uint CompSectorToDecomOffsetBitCount;       // 4 - bits -> 48 - 51 | P4
            public uint DecompSectorToCompSectorBitCount;      // 4 - bits -> 52 - 55 | P4
            public uint CRC;                                   // 4 - bits -> 56 - 59 |
            public uint Unknown;                               
            public uint ReadSectorSize;                        // 4 - bits -> 60 - 63 | P5
            public uint CompSectorSize;                        // 4 - bits -> 64 - 67 | P5
            public uint CompSectorCount;                       
            public uint FileSize;
            public uint FirstSectorPosition;

            public HeaderStruct(Stream s, int rev)
            {
                s.Seek(0, SeekOrigin.End);
                FileSize = (uint)s.Position;
                helper help = new helper();
                s.Seek(0, 0);
                MagicNumber = help.ReadU32(s);
                PackageVersion = help.RUInt32(s, rev);
                DecompressedFileSize = help.RUInt64(s, rev);
                Flags = help.RUInt32(s, rev);
                FileCount = help.RUInt32(s, rev);
                LocationCount = help.RUInt32(s, rev);
                HeaderSector = help.RUInt32(s, rev);
                FileSizeBitCount = help.RUInt32(s, rev);
                FileLocationCountBitCount = help.RUInt32(s, rev);
                FileLocationIndexBitCount = help.RUInt32(s, rev);
                LocationBitCount = help.RUInt32(s, rev);
                CompSectorToDecomOffsetBitCount = help.RUInt32(s, rev);
                DecompSectorToCompSectorBitCount = help.RUInt32(s, rev);
                CRC = help.RUInt32(s, rev);
                if (PackageVersion == 6)
                {
                    Unknown = help.ReadU32(s);
                    ReadSectorSize = 0x10000;
                    CompSectorSize = 0x4000;
                    HeaderSize = 64;
                }
                if (PackageVersion == 7)
                {
                    ReadSectorSize = (uint)help.RInt32(s, rev);
                    CompSectorSize = (uint)help.RInt32(s, rev);
                    HeaderSize = 72;
                }
                CompSectorCount = ((uint)CompSectorSize + FileSize - 1 - (uint)ReadSectorSize * HeaderSector) / (uint)CompSectorSize;
                FirstSectorPosition = (uint)(ReadSectorSize * HeaderSector) & 0xFFFF0000;
                if ((FirstSectorPosition % ReadSectorSize) != 0)
                    FirstSectorPosition += ReadSectorSize;
            }

            public static HeaderStruct ReadHeader(Stream s, int rev)
            {
                Header = new HeaderStruct(s, rev);
                return Header;
            }

        }
        public class SortedFileInfo
        {
            public uint SizeOf;
            public ulong dwHash = 0;
            public uint nSize = 0;
            public uint nLocationCount = 0;
            public uint nLocationIndex = 0;

            public SortedFileInfo()
            {
                SizeOf = 64;
                SizeOf += Header.FileSizeBitCount;
                SizeOf += Header.FileLocationCountBitCount;
                SizeOf += Header.FileLocationIndexBitCount;
                SizeOf *= Header.FileCount;
                SizeOf += 7;
                SizeOf /= 8;
            }

            public static SortedFileInfo[] Read_SortedFileInfo(Stream s)
            {
                SortedFileInfo[] result = new SortedFileInfo[Header.FileCount];
                helper help = new helper();
                SortedFileInfo sfi = new SortedFileInfo();
                s.Position = HeaderSize;
                byte[] Htable = ReturnBlock(s, sfi.SizeOf);
                uint position = 0;
                for (uint i = 0; i < Header.FileCount; i++)
                {
                    result[i] = new SortedFileInfo();
                    result[i].dwHash = help.ReadBits(Htable, position, 64);
                    position += 64;
                    result[i].nSize = (uint)help.ReadBits(Htable, position, Header.FileSizeBitCount);
                    position += Header.FileSizeBitCount;
                    result[i].nLocationCount = (uint)help.ReadBits(Htable, position, Header.FileLocationCountBitCount);
                    position += Header.FileLocationCountBitCount;
                    result[i].nLocationIndex = (uint)help.ReadBits(Htable, position, Header.FileLocationIndexBitCount);
                    position += Header.FileLocationIndexBitCount;
                }
                return result;
            }
        }

        public class Locations
        {
            public uint SizeOf;
            public uint index = 0;
            public ulong offset = 0;

            public Locations()
            {
                SizeOf = Header.LocationBitCount * Header.LocationCount;
                SizeOf += 7;
                SizeOf /= 8;
            }

            public static Locations[] Read_Locations(Stream s)
            {
                helper help = new helper();
                Locations l = new Locations();
                Locations[] result = new Locations[Header.LocationCount];
                uint[] index = new uint[Header.LocationCount];
                ulong[] offset = new ulong[Header.LocationCount];
                uint pos = 0;
                byte[] block = ReturnBlock(s, l.SizeOf);
                for (uint i = 0; i < Header.LocationCount; i++)
                {
                    offset[i] = help.ReadBits(block, pos, Header.LocationBitCount);
                    pos += Header.LocationBitCount;
                    index[i] = i;
                }
                Array.Sort(offset, index);
                for (uint i = 0; i < Header.LocationCount; i++)
                {
                    result[i] = new Locations();
                    result[i].index = index[i];
                    result[i].offset = offset[i];
                }
                return result;
            }
        }


        public class CompressedSectorToDecompressedOffset
        {
            public uint SizeOf;
            public uint SectorIndex = 0;
            public ulong DecompressedOffset = 0;

            public CompressedSectorToDecompressedOffset()
            {
                SizeOf = Header.CompSectorCount * Header.LocationBitCount;
                SizeOf += 7;
                SizeOf /= 8;
            }

            public static CompressedSectorToDecompressedOffset[] Read_CompressedSectorToDecompressedOffset(Stream s)
            {
                helper help = new helper();
                CompressedSectorToDecompressedOffset c = new CompressedSectorToDecompressedOffset();
                uint pos = 0;
                byte[] block = ReturnBlock(s, c.SizeOf);
                CompressedSectorToDecompressedOffset[] result = new CompressedSectorToDecompressedOffset[Header.CompSectorCount];
                for (uint i = 0; i < Header.CompSectorCount; i++)
                {
                    result[i] = new CompressedSectorToDecompressedOffset();
                    result[i].SectorIndex = i;
                    result[i].DecompressedOffset = help.ReadBits(block, pos, Header.LocationBitCount);
                    pos += Header.LocationBitCount;
                }
                return result;
            }

        }

        public class DecompressedSectorToCompressedSector
        {
            public uint SizeOf;
            public uint DecompressedSector = 0;
            public ulong CompressedSector = 0;

            public DecompressedSectorToCompressedSector()
            {
                helper help = new helper();
                SizeOf = help.GetHighestBit(Header.CompSectorCount) * (((uint)Header.DecompressedFileSize + (uint)Header.CompSectorSize - 1) / (uint)Header.CompSectorSize);
                SizeOf += 7;
                SizeOf /= 8;
            }
            public static DecompressedSectorToCompressedSector[] Read_DecompressedSectorToCompressedSector(Stream s)
            {
                helper help = new helper();
                uint temp = help.GetHighestBit(Header.CompSectorCount);
                DecompressedSectorToCompressedSector d = new DecompressedSectorToCompressedSector();
                uint sector = (((uint)Header.DecompressedFileSize + (uint)Header.CompSectorSize - 1) / (uint)Header.CompSectorSize);
                uint pos = 0;
                byte[] block = ReturnBlock(s, d.SizeOf);
                DecompressedSectorToCompressedSector[] result = new DecompressedSectorToCompressedSector[sector];

                for (uint i = 0; i < sector; i++)
                {
                    result[i] = new DecompressedSectorToCompressedSector();
                    result[i].DecompressedSector = i;
                    result[i].CompressedSector = help.ReadBits(block, pos, temp);
                    pos += temp;
                }
                return result;
            }

        }

        public class FileName
        {
            public uint offset = 0;
            public ulong fileHash = 0;
            public string filename = "";

            public static FileName[] Read_FileName(Stream s)
            {
                helper help = new helper();
                FileName[] result = new FileName[Header.FileCount];
                for (int i = 0; i < Header.FileCount; i++)
                {
                    FileName fn = new FileName();
                    if (Header.MagicNumber.ToString("X8").Equals("A1B2C3D4"))
                        fn.offset = help.ReadU32(s);
                    if (Header.MagicNumber.ToString("X8").Equals("D4C3B2A1"))           
                        fn.offset = help.ReverseUInt32(help.ReadU32(s));
                    result[i] = fn;
                }
                long pos = s.Position;
                for (int i = 0; i < Header.FileCount; i++)
                {
                    s.Seek(pos + result[i].offset, 0);
                    result[i].filename = help.ReadString(s);
                    result[i].fileHash = help.GetFileHash(result[i].filename);
                }
                return result;
            }
        }

        public class CompressedSectorChunk
        {
            public uint nr = 0;
            public uint position = 0;
            public ushort CompChunkSize = 0;
            public ushort DecompChunkSize = 0;
            public ushort flag = 0;
            public uint CompSector = 0;
            public ulong StartDecompOffset = 0;

            public static Dictionary<uint, CompressedSectorChunk> ReadSectors(Stream s)
            {
                helper help = new helper();
                CompressedSectorChunk csc = new CompressedSectorChunk();
                uint ChunkNumber = 0;
                ulong doffset = 0;
  
                s.Seek(Header.FirstSectorPosition, 0);
                Dictionary<uint, CompressedSectorChunk> result = new Dictionary<uint, CompressedSectorChunk>();               
                if (Header.MagicNumber.ToString("X8").Equals("A1B2C3D4"))
                {
                    help.ReadU16(s); help.ReadU16(s); s.Seek(help.ReadU16(s), SeekOrigin.Current);
                }
                if (Header.MagicNumber.ToString("X8").Equals("D4C3B2A1"))
                {
                    help.ReverseUInt16(help.ReadU16(s)); help.ReverseUInt16(help.ReadU16(s)); s.Seek(help.ReverseUInt16(help.ReadU16(s)), SeekOrigin.Current);
                }

                for (uint sector = 0; sector <= Header.CompSectorCount; sector++)
                {
                    uint SectorStartPosition = Header.FirstSectorPosition + sector * Header.CompSectorSize;
                    uint NextSectorPosition = SectorStartPosition + Header.CompSectorSize;
                    s.Seek(SectorStartPosition, 0);
                    while (s.Position + 0xf < NextSectorPosition)
                    {
                        csc = new CompressedSectorChunk();
                        csc.position = (uint)s.Position;
                        ushort Size = help.ReadU16(s);
                        ushort flag = help.ReadU16(s);
                        ushort CompChunkSize = help.ReadU16(s);
                        csc.nr = ChunkNumber;
                        csc.CompChunkSize = CompChunkSize;
                        csc.DecompChunkSize = Size;
                        csc.flag = flag;
                        csc.CompSector = sector;
                        doffset += Size;
                        csc.StartDecompOffset = doffset - Size;
                        result.Add(ChunkNumber, csc);
                        ChunkNumber++;
                        s.Seek(CompChunkSize, SeekOrigin.Current);
                    }
                }
                return result;
            }

            public static CompressedSectorChunk[] Read_CompressedSectorChunk(Dictionary<uint, CompressedSectorChunk> d)
            {
                uint count = 0;
                CompressedSectorChunk csc = new CompressedSectorChunk();
                CompressedSectorChunk[] result = new CompressedSectorChunk[d.Count];
                foreach (KeyValuePair<uint, cpk.CompressedSectorChunk> pair in d)
                {
                    csc = new CompressedSectorChunk();
                    csc = pair.Value;
                    result[count] = csc;
                    count++;
                }
                return result;
            }
        }

        public static HeaderStruct Header;
        public static uint HeaderSize = 64;

        public static int IsByteReverse(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            uint Magic = reader.ReadUInt32();
            if (Magic.ToString("X8").Equals("A1B2C3D4"))
                return 1;
            if (Magic.ToString("X8").Equals("D4C3B2A1"))
                return 0;
            else
                return -1;
        }

        public static byte[] ReturnBlock(Stream s, uint size)
        {
            byte[] block = new byte[size];
            s.Read(block, 0, (int)size);
            return block;
        }

    }
}
