using System;
using Xunit;

namespace CrispASR.Tests
{
    public class AuxiliaryNativeTests
    {
        [Fact]
        public void RegistryAndBooleanHelpersUseTheRealLibrary()
        {
            if (!Live.LibraryLoadable()) return;
            Assert.NotEmpty(ModelRegistry.Backends());
            Assert.Null(ModelRegistry.Lookup("__missing_binding_contract_backend__"));
            Assert.Null(ModelRegistry.DefaultBundle("__missing_binding_contract_backend__"));
            Assert.True(TextUtilities.KokoroLanguageIsGerman("de"));
            Assert.False(TextUtilities.KokoroLanguageIsGerman("en"));
            Assert.False(TextUtilities.KokoroLanguageHasNativeVoice("__unknown__"));
        }
    }
}
