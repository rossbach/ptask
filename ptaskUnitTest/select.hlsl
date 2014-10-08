//--------------------------------------------------------------------------------------
// File: select.hlsl
// This file contains a Compute Shader to perform select
// 1D array C = copy(A)
// maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------

#define INTSIZE 4
#define UPRC_DELTA ('A'-'a')
#define PFXSUM_BLOCK_SIZE  512

ByteAddressBuffer ki : register(t0);
ByteAddressBuffer vi : register(t1);
StructuredBuffer<int> kmi : register(t2);
StructuredBuffer<int> vmi : register(t3);
RWByteAddressBuffer ko : register(u0);
RWByteAddressBuffer vo : register(u1);
RWStructuredBuffer<int> kmo : register(u2);
RWStructuredBuffer<int> vmo : register(u3);
groupshared int g_shared[PFXSUM_BLOCK_SIZE];
groupshared int g_pin[PFXSUM_BLOCK_SIZE];

cbuffer cbCS : register( b0 )
{
    uint g_npairs;
};

inline bool islwr(int c) { return (c >= ((int)'a') && (c <= (int)'z')); }
inline bool isupr(int c) { return (c >= ((int)'A') && (c <= (int)'Z')); }
inline bool isalpha(int c) { return isupr(c) || islwr(c); }

/*! Return an upper-cased version of the input, 
 *  which is a 4-byte substring represented as an int
 *  \param uint c  
 *  \return upper-cased uint (4-byte substring)
 */
int toupr(uint c) {
    int c0 = (c >> 24) & 0xFF;
    int c1 = (c >> 16) & 0xFF;
    int c2 = (c >> 8) & 0xFF;
    int c3 = c & 0xFF;
    if(islwr(c0)) c0 += UPRC_DELTA;
    if(islwr(c1)) c1 += UPRC_DELTA;
    if(islwr(c2)) c2 += UPRC_DELTA;
    if(islwr(c3)) c3 += UPRC_DELTA;
    int res = ((c0 << 24) | (c1 << 16) | (c2 << 8) | c3);
    return res;
}

/*! Return the index in a 4-byte substring represented as an int
 *  at which the first '\0' character occurs, or -1 if it does
 *  not occur in the string. 
 *  \param uint c  
 *  \return index of '\0', or -1 if not present.
 */
int nullidx(
    int c
    )
{
    if((c & 0xFF) == 0) return 0;
    if((c & 0xFF00) == 0) return 1;
    if((c & 0xFF0000) == 0) return 2;
    if((c & 0xFF000000) == 0) return 3;
    return -1;
}

/*! Return true if the 4-byte substring represented as an int
 *  contains the '\0' character, false otherwise
 *  not occur in the string. 
 *  \param uint c  
 *  \return true if c contains '\0'
 */
inline bool hasterm(int c) { return nullidx(c) >= 0; }

/*! Compare characters, assuming each int contains
 *  a single byte of a string in the lsb. 
 *  \param int s, t
 *  \return ((s<t)?-1 :((s>t)?1:0))
 */
inline int charcmp(int s, int t) { return ((s<t)?-1 :((s>t)?1:0)); }

/*! Compare up to 4 characters in a 4-byte substring 
 *  given as an integer.
 *  \param int s, t
 *  \return ((s<t)?-1 :((s>t)?1:0))
 */
int substrcmp(
    int s,
    int t
    )
{
    int cres;
    int s0 = (s >> 24) & 0xFF;   int t0 = (t >> 24) & 0xFF;
    int s1 = (s >> 16) & 0xFF;   int t1 = (t >> 16) & 0xFF;
    int s2 = (s >> 8) & 0xFF;    int t2 = (t >> 8) & 0xFF;
    int s3 = s & 0xFF;           int t3 = t & 0xFF;    
    cres = charcmp(s3, t3);
    if(cres != 0 || (cres == 0 && (s3 == 0))) return cres;
    cres = charcmp(s2, t2);
    if(cres != 0 || (cres == 0 && (s2 == 0))) return cres;
    cres = charcmp(s1, t1);
    if(cres != 0 || (cres == 0 && (s1 == 0))) return cres;
    cres = charcmp(s0, t0);
    if(cres != 0 || (cres == 0 && (s0 == 0))) return cres;
    return 0;
}

/*! String compare, given byte addressable buffers
 *  containing each string, and offsets where the
 *  string begins in each buffer. Return same as 
 *  normal libc strcmp function. 
 *  \param ByteAddressBuffer strbase0
 *  \param ByteAddressBuffer strbase1
 *  \param int nOffset0
 *  \param int nOffset1
 *  \return ((s<t)?-1 :((s>t)?1:0))
 */
int strcmp(
    ByteAddressBuffer strbase0,
    ByteAddressBuffer strbase1,
    int nOffset0,
    int nOffset1
    )
{
    int off0 = nOffset0;
    int off1 = nOffset1;
    while(true) {
        int c0 = strbase0.Load(off0);
        int c1 = strbase1.Load(off1);
        if(hasterm(c0) || hasterm(c1))
            return substrcmp(c0, c1);
        if(c0 < c1) return -1;
        if(c0 > c1) return 1;
        off0 += 4;
        off1 += 4;
    }
    return 0;
}

/*! String length, given a byte addressable buffer
 *  and an offset where the string begins in the buffer. 
 *  Return same as  normal libc strlen function. 
 *  \param ByteAddressBuffer strbase
 *  \param int nOffset
 *  \return length of string starting at nOffset
 */
int strlen(
    ByteAddressBuffer strbase,
    int nOffset
    )
{
    int offset = nOffset;
    int nidx = -1;
    int len = 0;
    while(nidx < 0) {
        int c = strbase.Load(offset);
        nidx = nullidx(c);
        if(nidx < 0) 
            len += 4;
        else
            len += nidx;
        offset += 4;
    }
    return len;
}

/*! Arbitrary aligned store
 *  \param int c -- int to store
 *  \param RWByteAddressBuffer dst -- buffer to write result
 *  \param int nDstOffset -- offset of to write result
 *  \return number of bytes copied
 */
void
store(
    int c,
    RWByteAddressBuffer dst,
    int nDstOffset
    )
{
    int nPrevWord, nSubsqWord;
    int nModulus = nDstOffset % 4;
    int nBaseAddress = nDstOffset & ~0x3;
    if(nModulus == 0) {
    }
    switch(nModulus) {
    default:
    case 0: 
        dst.Store(nDstOffset, c);
        return;
    case 1:
        nPrevWord = dst.Load(nBaseAddress);
        nSubsqWord = dst.Load(nBaseAddress+4);
        nPrevWord &= 0xFF;
        nPrevWord |= ((c << 8) & 0xFFFFFF00);
        nSubsqWord &= 0xFFFFFF00;
        nSubsqWord |= ((c >> 24) & 0xFF);
        dst.Store(nBaseAddress, nPrevWord);
        dst.Store(nBaseAddress+4, nSubsqWord);
        break;
    case 2:
        nPrevWord = dst.Load(nBaseAddress);
        nSubsqWord = dst.Load(nBaseAddress+4);
        nPrevWord &= 0xFFFF;
        nPrevWord |= ((c << 16) & 0xFFFF0000);
        nSubsqWord &= 0xFFFF00;
        nSubsqWord |= ((c >> 16) & 0xFFFF);
        dst.Store(nBaseAddress, nPrevWord);
        dst.Store(nBaseAddress+4, nSubsqWord);
        break;
    case 3:
        nPrevWord = dst.Load(nBaseAddress);
        nSubsqWord = dst.Load(nBaseAddress+4);
        nPrevWord &= 0xFFFFFF;
        nPrevWord |= ((c << 24) & 0xFF000000);
        nSubsqWord &= 0xFF00;
        nSubsqWord |= ((c >> 8) & 0xFFFFFF);
        dst.Store(nBaseAddress, nPrevWord);
        dst.Store(nBaseAddress+4, nSubsqWord);
        break;
    }
}

/*! Copy src to dst. 
 *  Src is assumed to be 4-byte aligned.
 *  Dst is not assumed to be 4-byte aligned, but it is 
 *  assumed safe to assume write 4 bytes at the tail of the
 *  string.
 *  \param ByteAddressBuffer src -- buffer containing input string
 *  \param int nSrcOffset -- offset of source string
 *  \param RWByteAddressBuffer dst -- buffer to write result
 *  \param int nDstOffset -- offset of to write result
 *  \return number of bytes copied
 */
int strcpy(
    ByteAddressBuffer src,
    int nSrcOffset,
    RWByteAddressBuffer dst,
    int nDstOffset
    )
{
    int nidx = -1;
    int nWritten = 0;
    int nCurDstOffset = nDstOffset;
    int nCurSrcOffset = nSrcOffset;
    while(nidx < 0) {
        int c = asint(src.Load(nCurSrcOffset));
        store(c, dst, nCurDstOffset); 
        nidx = nullidx(c);
        nCurDstOffset += 4;
        nCurSrcOffset += 4;
        nWritten += ((nidx < 0) ? 4:nidx);
    }
    return nWritten;
}

/*! Copy src to dst, up to a given limit.
 *  Src is assumed to be 4-byte aligned.
 *  Dst is not assumed to be 4-byte aligned, but it is 
 *  assumed safe to assume write 4 bytes at the tail of the
 *  string.
 *  \param ByteAddressBuffer src -- buffer containing input string
 *  \param int nSrcOffset -- offset of source string
 *  \param RWByteAddressBuffer dst -- buffer to write result
 *  \param int nDstOffset -- offset of to write result
 *  \param int nMaxChars -- maximum characters to write.
 *  \return number of bytes copied
 */
int strncpy(
    ByteAddressBuffer src,
    int nSrcOffset,
    RWByteAddressBuffer dst,
    int nDstOffset,
    int nMaxChars
    )
{
    int nidx = -1;
    int nWritten = 0;
    int nCurOffset = nDstOffset;
    while(nidx < 0 && nWritten < nMaxChars) {
        int c = asint(src.Load(nSrcOffset));
        dst.Store(nCurOffset, asint(c));
        nidx = nullidx(c);
        nCurOffset += 4;
        nWritten += ((nidx < 0) ? 4:nidx);
    }
    return nWritten;
}


/*! Given two strings, concatenate them!
 *  \param ByteAddressBuffer strsrc0 -- buffer containing input string 0
 *  \param int nOffset0 -- offset of source string 0 in strsrc0
 *  \param ByteAddressBuffer strsrc1 -- buffer containing input string 1
 *  \param int nOffset1 -- offset of source string 1 in strsrc1
 *  \param RWByteAddressBuffer strdst -- buffer to write result
 *  \param int nDstOffset -- offset of to write result
 *  \return length of resulting new string
 */
int strcat(
    ByteAddressBuffer strsrc0,
    int nOffset0,
    ByteAddressBuffer strsrc1,
    int nOffset1,
    RWByteAddressBuffer strdst,
    int nDstOffset
    )
{
    int nsrclen0 = strlen(strsrc0, nOffset0);
    int nsrclen1 = strlen(strsrc1, nOffset1);
    strcpy(strsrc0, nOffset0, strdst, nDstOffset);
    strcpy(strsrc1, nOffset1, strdst, nDstOffset+nsrclen0);
    return nsrclen0 + nsrclen1;
}

/*! 
 * Given two strings, predict the buffer size required to hold
 *  the string that results from concatenating them. Note that
 *  the last two parameters are unused, but we keep them to 
 *  preserve the signature of strcat.
 * 
 *  \param ByteAddressBuffer strsrc0 -- buffer containing input string 0
 *  \param int nOffset0 -- offset of source string 0 in strsrc0
 *  \param ByteAddressBuffer strsrc1 -- buffer containing input string 1
 *  \param int nOffset1 -- offset of source string 1 in strsrc1
 *  \param RWByteAddressBuffer strdst -- buffer to write result
 *  \param int nDstOffset -- offset of to write result
 *  \return predicted length of resulting new string, padded to 4-byte aligned.
 */
int strcat_outsize(
    ByteAddressBuffer strsrc0,
    int nOffset0,
    ByteAddressBuffer strsrc1,
    int nOffset1,
    RWByteAddressBuffer strdst,
    int nDstOffset
    )
{
    int reslen = 
        strlen(strsrc0, nOffset0) +
        strlen(strsrc1, nOffset1);
    reslen += (4 - (reslen % 4));
    return reslen;
}

/*! Compute the prefix sum given an 
 *  array of integers as input
 *  \param int idx -- thread id
 *  \param int N -- number of elements in input
 *  \param ByteAddressBuffer pin -- input array
 *  \param RWByteAddressBuffer pout -- output array
 *  \return void (output written to pout parameter)
 */
void pfxsum( 
    int idx,
    int N,
    RWByteAddressBuffer pout
    ) 
{
    int d, e;
    int offset = 1;
    g_shared[idx*2] = g_pin[2*idx];
    g_shared[2*idx+1] = g_pin[2*idx+1];
    for(d = N >> 1; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();
        if(idx < d) {
            int ai = offset*(2*idx+1)-1;
            int bi = offset*(2*idx+2)-1;
            g_shared[bi] += g_shared[ai];
        }
        offset *= 2;
    }
    if(idx == 0) {
        g_shared[N-1] = 0;
    }
    for(e=1; e<N; e*=2) {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if(idx < e) {
            int ai = offset*(2*idx+1)-1;   
            int bi = offset*(2*idx+2)-1;  
            int t = g_shared[ai];   
            g_shared[ai] = g_shared[bi];   
            g_shared[bi] += t; 
        }
    }
    GroupMemoryBarrierWithGroupSync();
    pout.Store((2*idx*INTSIZE), g_shared[2*idx]);
    pout.Store(((2*idx+1)*INTSIZE), g_shared[2*idx+1]);
}

/*! generic predicate. 
 *  Ideally this becomes autogenerated code. *****
 *  \param  int idx -- thread id
 *  \param  int N -- number of elements in input
 *  \param  ByteAddressBuffer keys 
 *  \param  ByteAddressBuffer values,
 *  \param  StructuredBuffer<int> keymap,
 *  \param  StructuredBuffer<int> valuemap,
 *  \return void (output written to pout parameter)
 */
bool 
predicate(
    int idx,
    int N,
    ByteAddressBuffer keys,
    ByteAddressBuffer values,
    StructuredBuffer<int> keymap,
    StructuredBuffer<int> valuemap
    )
{
    // mimic auto-generated code for 
    // LEN(key) == len(value). Note we
    // cannot use the length value from
    // the key map because we want the actual
    // string length, rather than the length
    // allocated to the string. 
    uint nKInputOffset = kmi[idx];
    uint nVInputOffset = vmi[idx];
    int keylen = strlen(keys, nKInputOffset);
    int vallen = strlen(values, nVInputOffset);
    return keylen == vallen;
}

// note that the number of thread-groups has 
// currently to be defined to match the number 
// of pairs in the constant buffer. Ugh. 
#define thread_group_size_x 32
#define thread_group_size_y 1
#define thread_group_size_z 1

/*! 
 * Compute SELECT STRCAT(k1, v1) where LEN(k1) == LEN(k2)
 *  \param uint3 DTid -- SV_DispatchThreadID 
 *  \return void (output written to pout parameter)
 */
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    uint i;
	int idx	= DTid.x * 2;
    uint nKInputOffset = kmi[idx];
    uint nKInputLength = kmi[idx+1];
    uint nKInputOffsetMax = nKInputOffset + (nKInputLength + (4-(nKInputLength%4)));
    uint nVInputOffset = vmi[idx];
    uint nVInputLength = vmi[idx+1];
    uint nVInputOffsetMax = nVInputOffset + (nVInputLength + (4-(nVInputLength%4)));
    for(i=nKInputOffset; i<nKInputOffsetMax; i+=4) {
        ko.Store(i, toupr(asint(ki.Load(i))));
    }
    int nBufSize = 0;
    vo.Store(DTid.x*INTSIZE, 0); // clear expliclity
    bool bPredicatePass = predicate(idx, g_npairs, ki, vi, kmi, vmi);
    GroupMemoryBarrierWithGroupSync();
    if(bPredicatePass)
        nBufSize = strcat_outsize(ki, nKInputOffset, vi, nVInputOffset, vo, nVInputOffset);
    g_pin[DTid.x] = nBufSize;
	kmo[idx] = nKInputOffset;
	kmo[idx+1] = nKInputLength;
    GroupMemoryBarrierWithGroupSync();
    pfxsum(DTid.x, g_npairs, vo);    
    GroupMemoryBarrierWithGroupSync();
    int nOutputOffset = vo.Load(DTid.x*INTSIZE);
	vmo[idx] = nOutputOffset;
	vmo[idx+1] = nBufSize;
    GroupMemoryBarrierWithGroupSync();
    if(bPredicatePass)
        strcat(ki, nKInputOffset, vi, nVInputOffset, vo, nOutputOffset);
    // strcpy(ki, nKInputOffset, vo, nOutputOffset);
}

/*! 
 * Compute SELECT STRCAT(k1, v1) 
 *  \param uint3 DTid -- SV_DispatchThreadID 
 *  \return void (output written to pout parameter)
 */
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void select_strcat( uint3 DTid : SV_DispatchThreadID )
{
    uint i;
	int idx	= DTid.x * 2;
    uint nKInputOffset = kmi[idx];
    uint nKInputLength = kmi[idx+1];
    uint nKInputOffsetMax = nKInputOffset + (nKInputLength + (4-(nKInputLength%4)));
    uint nVInputOffset = vmi[idx];
    uint nVInputLength = vmi[idx+1];
    uint nVInputOffsetMax = nVInputOffset + (nVInputLength + (4-(nVInputLength%4)));
    for(i=nKInputOffset; i<nKInputOffsetMax; i+=4) {
        ko.Store(i, toupr(asint(ki.Load(i))));
    }
    int nBufSize = strcat_outsize(ki, nKInputOffset, vi, nVInputOffset, vo, nVInputOffset);
    g_pin[DTid.x] = nBufSize;
	kmo[idx] = nKInputOffset;
	kmo[idx+1] = nKInputLength;
    GroupMemoryBarrierWithGroupSync();
    pfxsum(DTid.x, g_npairs, vo);    
    GroupMemoryBarrierWithGroupSync();
    int nOutputOffset = vo.Load(DTid.x*INTSIZE);
	vmo[idx] = nOutputOffset;
	vmo[idx+1] = g_pin[DTid.x] = nBufSize;
    GroupMemoryBarrierWithGroupSync();
    strcat(ki, nKInputOffset, vi, nVInputOffset, vo, nOutputOffset);
    // strcpy(ki, nKInputOffset, vo, nOutputOffset);
}

/*! 
 * Compute SELECT UPPER(k1), v1 where k1 = 2nd entry
 *  \param uint3 DTid -- SV_DispatchThreadID 
 *  \return void (output written to pout parameter)
 */
[numthreads(thread_group_size_x, thread_group_size_y, thread_group_size_z)]
void select_toupr_implicit_pred( uint3 DTid : SV_DispatchThreadID )
{
    uint i;
	int idx	= DTid.x * 2;
    uint nKInputOffset = kmi[idx];
    uint nKInputLength = kmi[idx+1];
    uint nKInputOffsetMax = nKInputOffset + (nKInputLength + (4-(nKInputLength%4)));
    uint nVInputOffset = vmi[idx];
    uint nVInputLength = vmi[idx+1];
    uint nVInputOffsetMax = nVInputOffset + (nVInputLength + (4-(nVInputLength%4)));
    if(!strcmp(ki, ki, nKInputOffset, 8)) {
        for(i=nKInputOffset; i<nKInputOffsetMax; i+=4) {
            ko.Store(i, toupr(asint(ki.Load(i))));
        }
    }
    for(i=nVInputOffset; i<nVInputOffsetMax; i+=4) {
        vo.Store(i, toupr(asint(vi.Load(i))));
    }
	kmo[idx] = nKInputOffset;
	kmo[idx+1] = nKInputLength;
	vmo[idx] = nVInputOffset;
	vmo[idx+1] = nVInputLength;
}


