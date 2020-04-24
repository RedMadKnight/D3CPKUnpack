using System;
using System.Collections.Generic;
using System.IO;

namespace D3CPKUnpack
{
    class cpk
    {
        public class HeaderStruct
        {
            public uint MagicNumber;
            public uint PackageVersion;
            public ulong DecompressedFileSize;
            public uint Flags;
            public uint FileCount;
            public uint LocationCount;
            public uint HeaderSector;
            public uint FileSizeBitCount;
            public uint FileLocationCountBitCount;
            public uint FileLocationIndexBitCount;
            public uint LocationBitCount;
            public uint CompSectorToDecomOffsetBitCount;
            public uint DecompSectorToCompSectorBitCount;
            public uint CRC;
            public uint Unknown;
            public uint ReadSectorSize;
            public uint CompSectorSize;
            public uint CompSectorCount;
            public uint FileSize;

            public HeaderStruct(Stream s, int rev)
            {
                s.Seek(0, SeekOrigin.End);
                FileSize = (uint)s.Position;
                helper help = new helper();
                s.Seek(0, 0);
                MagicNumber = help.RUInt32(s, rev);
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
                    result[i].dwHash = (uint)help.ReadBits(Htable, position, 64);
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
                        fn.offset = help.ReverseUInt32(help.ReadU32(s));
                    if (Header.MagicNumber.ToString("X8").Equals("D4C3B2A1"))
                        fn.offset = help.ReadU32(s);
                    result[i] = fn;
                }
                long pos = s.Position;
                for (int i = 0; i < Header.FileCount; i++)
                {
                    s.Seek(pos + result[i].offset, 0);
                    result[i].filename = help.ReadString(s);
                    result[i].fileHash = help.GetFileHash(result[i].filename);
                    break;
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

            public static Dictionary<uint, CompressedSectorChunk> ReadSectors(Stream s)
            {
                helper help = new helper();
                CompressedSectorChunk csc = new CompressedSectorChunk();
                uint a = 0;
                uint pos = (uint)s.Position & 0xFFFF0000;
                if ((s.Position % 0x10000) != 0)
                    pos += 0x10000;
                uint sector = 1;
                uint next_sector = pos + 0x4000;
                Dictionary<uint, CompressedSectorChunk> result = new Dictionary<uint, CompressedSectorChunk>();
                pos += 6; //skip first empty record
                s.Seek(pos, 0); 
                while (true)
                {
                    pos = (uint)s.Position;
                    if (pos + 0xf > next_sector)
                    {
                        pos = next_sector;
                        next_sector += 0x4000;
                        s.Seek(pos, 0);
                        sector++;
                        if (next_sector > s.Length)
                            break;
                    }
                    a++;
                    ushort Size = help.ReadU16(s);
                    ushort flag = help.ReadU16(s);
                    ushort ComSectorSize = help.ReadU16(s);
                    if (ComSectorSize == 0)
                        break;
                    csc = new CompressedSectorChunk();
                    csc.nr = a;
                    csc.position = pos;
                    csc.CompChunkSize = ComSectorSize;
                    csc.DecompChunkSize = Size;
                    csc.flag = flag;
                    csc.CompSector = sector;
                    result.Add(pos, csc);
                    s.Seek(ComSectorSize, SeekOrigin.Current);
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
