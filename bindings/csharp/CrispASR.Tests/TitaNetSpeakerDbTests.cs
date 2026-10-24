using System;
using Xunit;
using CrispASR;

namespace CrispASR.Tests
{
    /// <summary>
    /// Tests for TitaNet and SpeakerDb types. Live tests are gated
    /// by the availability of the native library and model files.
    /// </summary>
    public class TitaNetTests
    {
        // Routed through Live so CRISPASR_CS_REQUIRE_LIVE can turn a silent
        // skip into a failure (issue #291 — nothing in CI ever loaded the
        // library, so the suite was green either way).
        private static bool CanLoadLibrary() => Live.LibraryLoadable();

        [Fact]
        public void TitaNet_ImplementsIDisposable()
        {
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(TitaNet)));
        }

        [Fact]
        public void TitaNet_Open_ThrowsOnBadPath()
        {
            if (!CanLoadLibrary()) return;
            Assert.Throws<InvalidOperationException>(() =>
                TitaNet.Open("/nonexistent/titanet.gguf"));
        }

        [Fact]
        public void CosineSim_IdenticalVectors_ReturnsOne()
        {
            if (!CanLoadLibrary()) return;
            var v = new float[] { 1, 0, 0, 0 };
            float sim = TitaNet.CosineSim(v, v);
            Assert.InRange(sim, 0.999f, 1.001f);
        }

        [Fact]
        public void CosineSim_OrthogonalVectors_ReturnsZero()
        {
            if (!CanLoadLibrary()) return;
            var a = new float[] { 1, 0, 0, 0 };
            var b = new float[] { 0, 1, 0, 0 };
            float sim = TitaNet.CosineSim(a, b);
            Assert.InRange(sim, -0.001f, 0.001f);
        }

        [Fact]
        public void CosineSim_OppositeVectors_ReturnsNegOne()
        {
            if (!CanLoadLibrary()) return;
            var a = new float[] { 1, 0, 0, 0 };
            var b = new float[] { -1, 0, 0, 0 };
            float sim = TitaNet.CosineSim(a, b);
            Assert.InRange(sim, -1.001f, -0.999f);
        }

        [Fact]
        public void CosineSim_MismatchedDimensions_Throws()
        {
            var a = new float[] { 1, 0 };
            var b = new float[] { 1, 0, 0 };
            Assert.Throws<ArgumentException>(() => TitaNet.CosineSim(a, b));
        }
    }

    public class SpeakerDbTests
    {
        [Fact]
        public void SpeakerDb_ImplementsIDisposable()
        {
            Assert.True(typeof(IDisposable).IsAssignableFrom(typeof(SpeakerDb)));
        }

        [Theory]
        [InlineData("Alice")]
        [InlineData("")]
        [InlineData(" \t\r\n")]
        [InlineData(null)]
        public void SpeakerDb_Open_ThrowsWithoutConsent(string? expectedNames)
        {
            // Consent is checked before argument validation or native access.
            Assert.Throws<InvalidOperationException>(() =>
                SpeakerDb.Open("/nonexistent/speakers", expectedNames!, consentAttested: false));
        }

        [Theory]
        [InlineData("")]
        [InlineData(" ")]
        [InlineData("\t\r\n")]
        public void SpeakerDb_Open_ThrowsOnEmptyOrWhitespaceRoster(string expectedNames)
        {
            // Closed-roster validation is managed-side; no native library needed.
            var error = Assert.Throws<ArgumentException>(() =>
                SpeakerDb.Open("/nonexistent/speakers", expectedNames, consentAttested: true));
            Assert.Equal("expectedNames", error.ParamName);
        }

        [Fact]
        public void SpeakerDb_Open_ThrowsOnNullRoster()
        {
            var error = Assert.Throws<ArgumentNullException>(() =>
                SpeakerDb.Open("/nonexistent/speakers", null!, consentAttested: true));
            Assert.Equal("expectedNames", error.ParamName);
        }

        [Fact]
        public void SpeakerDb_Open_ThrowsOnNullDirectory()
        {
            var error = Assert.Throws<ArgumentNullException>(() =>
                SpeakerDb.Open(null!, "Alice", consentAttested: true));
            Assert.Equal("dirPath", error.ParamName);
        }

        [Fact]
        public void SpeakerDb_Enroll_ThrowsWithoutConsent()
        {
            Assert.Throws<InvalidOperationException>(() =>
                SpeakerDb.Enroll("/nonexistent/speakers", "alice", new float[] { 1, 0 }, consentAttested: false));
        }
    }
}
