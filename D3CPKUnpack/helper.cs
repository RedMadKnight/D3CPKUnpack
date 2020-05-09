using System;
using System.IO;
using System.Linq;
using System.Text;
using ICSharpCode.SharpZipLib.Zip.Compression.Streams;


namespace D3CPKUnpack
{
    public class helper
    {
        public uint GetHighestBit(uint u)
        {
            uint result = 0;
            while (u != 0)
            {
                u = (u >> 1);
                result++;
            }
            return result;
        }

        public string ReadString(Stream s)
        {
            string result = "";
            char b;
            while ((b = (char)s.ReadByte()) != (char)0)
                result += b;
            return result;
        }

        public byte ReadU8(Stream s)
        {
            return (byte)s.ReadByte();
        }

        public ushort ReadU16(Stream s)
        {
            ushort res = 0;
            res |= (byte)s.ReadByte();
            res = (ushort)((res << 8) | (byte)s.ReadByte());
            return res;
        }
        public uint ReadU32(Stream s)
        {
            uint res = 0;
            res |= (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            return res;
        }

        public ulong ReadU64(Stream s)
        {
            ulong res = 0;
            res |= (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            res = (res << 8) | (byte)s.ReadByte();
            return res;
        }

        public Int16 ReadBinaryInt16(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadInt16();
        }
        public UInt16 ReadBinaryUInt16(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadUInt16();
        }

        public Int32 ReadBinaryInt32(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadInt32();
        }

        public UInt32 ReadBinaryUInt32(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadUInt32();
        }

        public Int64 ReadBinaryInt64(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadInt64();
        }

        public UInt64 ReadBinaryUInt64(Stream s)
        {
            BinaryReader reader = new BinaryReader(s);
            return reader.ReadUInt64();
        }

        public ulong ReadBits(byte[] buff, uint bitPos, uint bitCount)
        {
            ulong result = 0;
            for (uint i = 0; i < bitCount; i++)
            {
                uint pos = bitPos + i;
                uint bytePos = pos / 8;
                uint byteBit = 7 - pos % 8;
                result = result << 1;
                if ((buff[bytePos] & (1 << (int)byteBit)) != 0)
                    result |= 1;
            }
            return result;
        }
        public string ReverseString(string source)
        {
            char[] dest = source.ToArray();
            string result = "";
            for (int i = dest.Length - 1; i >= 0; i--)
                result += dest[i];
            return result;
        }

        public Int16 ReverseInt16(Int16 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToInt16(bytes, 0);
            return x;
        }

        public UInt16 ReverseUInt16(UInt16 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToUInt16(bytes, 0);
            return x;
        }

        public Int32 ReverseInt32(Int32 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToInt32(bytes, 0);
            return x;
        }

        public UInt32 ReverseUInt32(UInt32 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToUInt32(bytes, 0);
            return x;
        }

        public Int64 ReverseInt64(Int64 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToInt64(bytes, 0);
            return x;
        }

        public UInt64 ReverseUInt64(UInt64 x)
        {
            byte[] bytes = BitConverter.GetBytes(x);
            Array.Reverse(bytes);
            x = BitConverter.ToUInt64(bytes, 0);
            return x;
        }

        public Int16 RInt16(Stream s, int r)
        {
            if (r == 0)
                return ReverseInt16(ReadBinaryInt16(s));
            else
                return ReadBinaryInt16(s);
        }

        public UInt16 RUInt16(Stream s, int r)
        {
            if (r == 0)
                return ReverseUInt16(ReadBinaryUInt16(s));
            else
                return ReadBinaryUInt16(s);
        }

        public Int32 RInt32(Stream s, int r)
        {
            if (r == 0)
                return ReverseInt32(ReadBinaryInt32(s));
            else
                return ReadBinaryInt32(s);
        }

        public UInt32 RUInt32(Stream s, int r)
        {
            if (r == 0)
                return ReverseUInt32(ReadBinaryUInt32(s));
            else
                return ReadBinaryUInt32(s);
        }

        public Int64 RInt64(Stream s, int r)
        {
            if (r == 0)
                return ReverseInt64(ReadBinaryInt64(s));
            else
                return ReadBinaryInt64(s);
        }

        public UInt64 RUInt64(Stream s, int r)
        {
            if (r == 0)
                return ReverseUInt64(ReadBinaryUInt64(s));
            else
                return ReadBinaryUInt64(s);
        }

        public string RString(Stream s, int r)
        {
            if (r == 0)
                return ReverseString(ReadString(s));
            else
                return ReadString(s);
        }

        public ulong Hash64(string name)
        {
            var bytes = Encoding.ASCII.GetBytes(name);
            ulong result = 0xCBF29CE484222325L;
            for (int i = 0; i < name.Length; i++)
                result = 0x100000001B3L * (result ^ bytes[i]);
            return result;
        }

        public ulong Hash64More(string name, ulong previousHash)
        {
            var bytes = Encoding.ASCII.GetBytes(name);
            for (int i = 0; i < name.Length; i++)
                previousHash = 0x100000001B3L * (bytes[i] ^ previousHash);
            return previousHash;
        }

        public ulong GetSubFileHash(ulong dwParentHash, string szFilename)
        {
            return Hash64More(szFilename.ToLower(), dwParentHash);
        }

        public ulong GetFileHash(string szFilename)
        {
            return Hash64(szFilename.ToLower());
        }

        public byte[] AddByteArray(byte[] source1, byte[] source2)
        {
            int newSize = source1.Length + source2.Length;
            var ms = new MemoryStream(new byte[newSize], 0, newSize, true, true);
            ms.Write(source1, 0, source1.Length);
            ms.Write(source2, 0, source2.Length);
            byte[] result = ms.GetBuffer();
            return result;
        }

        public byte[] DecompressZlib(byte[] input)
        {
            MemoryStream source = new MemoryStream(input);
            byte[] result = null;
            using (MemoryStream outStream = new MemoryStream())
            {
                using (InflaterInputStream inf = new InflaterInputStream(source))
                {
                    inf.CopyTo(outStream);
                }
                result = outStream.ToArray();
            }
            return result;
        }

        public byte[] DecompressChunk(Stream fs, int offset, int rev)
        {
            //CPK_MAX_DECOMP_BUFFER_SIZE = 0x10000
            helper help = new helper();
            fs.Seek(offset, 0);
            uint DecompressedSize, Flag, CompressedSize;
            if (rev == 0)
            {
                DecompressedSize = help.ReadU16(fs);
                Flag = help.ReadU16(fs);
                CompressedSize = help.ReadU16(fs);
            }
            else
            {
                DecompressedSize = help.ReverseUInt16(help.ReadU16(fs));
                Flag = help.ReverseUInt16(help.ReadU16(fs));
                CompressedSize = help.ReverseUInt16(help.ReadU16(fs));
            }
            byte[] buff = new byte[CompressedSize];
            fs.Read(buff, 0, (int)CompressedSize);
            byte[] tmp = { };
            try
            {
                tmp = DecompressZlib(buff);
            }
            catch
            {
                tmp = buff;
            }
            return tmp;
        }
    }
}
