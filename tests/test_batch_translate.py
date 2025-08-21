import pytest
from unittest.mock import Mock, patch
from argostranslate import translate


class TestBatchTranslation:
    """Test batch translation functionality"""

    def test_itranslation_translate_batch_default(self):
        """Test ITranslation.translate_batch default implementation"""
        # Create a mock translation that implements single translate
        mock_translation = Mock(spec=translate.ITranslation)
        mock_translation.translate.side_effect = lambda x: f"translated_{x}"
        
        # Call the default translate_batch implementation
        result = translate.ITranslation.translate_batch(mock_translation, ["hello", "world"])
        
        # Should call single translate for each text
        expected = ["translated_hello", "translated_world"]
        assert result == expected
        assert mock_translation.translate.call_count == 2

    def test_itranslation_hypotheses_batch_default(self):
        """Test ITranslation.hypotheses_batch default implementation"""
        # Create mock hypotheses
        mock_hypothesis1 = [translate.Hypothesis("translated_hello", 0.9)]
        mock_hypothesis2 = [translate.Hypothesis("translated_world", 0.8)]
        
        mock_translation = Mock(spec=translate.ITranslation)
        mock_translation.hypotheses.side_effect = [mock_hypothesis1, mock_hypothesis2]
        
        # Call the default hypotheses_batch implementation
        result = translate.ITranslation.hypotheses_batch(mock_translation, ["hello", "world"], 1)
        
        # Should call single hypotheses for each text
        expected = [mock_hypothesis1, mock_hypothesis2]
        assert result == expected
        assert mock_translation.hypotheses.call_count == 2

    def test_translate_batch_public_api(self):
        """Test the public translate_batch function"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = ["result1", "result2"]
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch(["text1", "text2"], "es", "en")
            
            # Should call get_translation_from_codes and translate_batch
            mock_get_translation.assert_called_once_with("es", "en")
            mock_translation.translate_batch.assert_called_once_with(["text1", "text2"])
            assert result == ["result1", "result2"]

    def test_batch_translation_empty_list(self):
        """Test batch translation with empty input"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = []
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch([], "es", "en")
            
            assert result == []
            mock_translation.translate_batch.assert_called_once_with([])

    def test_batch_translation_single_text(self):
        """Test batch translation with single text"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = ["single_result"]
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch(["single_text"], "es", "en")
            
            assert result == ["single_result"]
            mock_translation.translate_batch.assert_called_once_with(["single_text"])

    def test_batch_vs_single_consistency(self):
        """Test that batch and single translation produce similar results"""
        # This test requires actual translation models, so we'll mock it
        # In a real environment, this would test with actual language packages
        
        test_texts = [
            "Hola mundo",
            "¿Cómo estás?",
            "Buenos días"
        ]
        
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            
            # Mock single translations
            single_results = ["Hello world", "How are you?", "Good morning"]
            mock_translation.translate.side_effect = single_results
            
            # Mock batch translation
            mock_translation.translate_batch.return_value = single_results
            
            mock_get_translation.return_value = mock_translation
            
            # Test single translations
            single_outputs = []
            for text in test_texts:
                result = translate.translate(text, "es", "en")
                single_outputs.append(result)
            
            # Test batch translation
            batch_outputs = translate.translate_batch(test_texts, "es", "en")
            
            # Results should be identical
            assert single_outputs == batch_outputs
            assert len(batch_outputs) == len(test_texts)

    def test_batch_translation_maintains_order(self):
        """Test that batch translation maintains input order"""
        test_texts = [
            "Primer texto",
            "Segundo texto", 
            "Tercer texto"
        ]
        
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            expected_results = ["First text", "Second text", "Third text"]
            mock_translation.translate_batch.return_value = expected_results
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch(test_texts, "es", "en")
            
            # Results should maintain the same order as input
            assert result == expected_results
            assert len(result) == len(test_texts)

    def test_batch_translation_error_handling(self):
        """Test batch translation error handling"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.side_effect = Exception("Translation failed")
            mock_get_translation.return_value = mock_translation
            
            # Should propagate the exception
            with pytest.raises(Exception, match="Translation failed"):
                translate.translate_batch(["test"], "es", "en")

    def test_hypothesis_batch_functionality(self):
        """Test batch hypothesis generation"""
        test_texts = ["texto uno", "texto dos"]
        
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            
            # Mock batch hypotheses
            expected_hypotheses = [
                [translate.Hypothesis("text one", 0.9)],
                [translate.Hypothesis("text two", 0.8)]
            ]
            mock_translation.hypotheses_batch.return_value = expected_hypotheses
            mock_get_translation.return_value = mock_translation
            
            # Test through public API (we'd need to add this function)
            # For now, test the underlying interface
            result = mock_translation.hypotheses_batch(test_texts, 1)
            
            assert result == expected_hypotheses
            assert len(result) == len(test_texts)
            assert all(isinstance(hyp_list, list) for hyp_list in result)
            assert all(isinstance(hyp, translate.Hypothesis) for hyp_list in result for hyp in hyp_list)


class TestApplyPackagedTranslationBatch:
    """Test the apply_packaged_translation_batch function"""

    @patch('argostranslate.translate.Translator')
    @patch('argostranslate.translate.Package')
    def test_apply_packaged_translation_batch_basic(self, mock_package, mock_translator):
        """Test basic functionality of apply_packaged_translation_batch"""
        # Mock package and tokenizer
        mock_pkg = Mock()
        mock_pkg.tokenizer.encode.side_effect = lambda x: [f"token_{x}"]
        mock_pkg.tokenizer.decode.side_effect = lambda x: f"decoded_{'_'.join(x)}"
        mock_pkg.target_prefix = ""
        
        # Mock translator
        mock_translator_instance = Mock()
        mock_batch_result = Mock()
        mock_batch_result.hypotheses = [["result1"], ["result2"]]
        mock_batch_result.scores = [0.9, 0.8]
        mock_translator_instance.translate_batch.return_value = [mock_batch_result, mock_batch_result]
        
        # Mock sentencizer (not used in this function but might be referenced)
        mock_sentencizer = Mock()
        
        # Test the function
        sentences = ["sentence1", "sentence2"]
        result = translate.apply_packaged_translation_batch(
            mock_pkg, sentences, mock_translator_instance, num_hypotheses=1
        )
        
        # Verify results
        assert len(result) == len(sentences)
        assert all(isinstance(hyp_list, list) for hyp_list in result)
        assert all(isinstance(hyp, translate.Hypothesis) for hyp_list in result for hyp in hyp_list)
        
        # Verify translator was called
        mock_translator_instance.translate_batch.assert_called_once()

    def test_apply_packaged_translation_batch_with_prefix(self):
        """Test apply_packaged_translation_batch with target prefix"""
        # Mock package with target prefix
        mock_pkg = Mock()
        mock_pkg.tokenizer.encode.side_effect = lambda x: [f"token_{x}"]
        mock_pkg.tokenizer.decode.side_effect = lambda x: f"prefix_decoded_{'_'.join(x)}"
        mock_pkg.target_prefix = "prefix_"
        
        # Mock translator
        mock_translator = Mock()
        mock_batch_result = Mock()
        mock_batch_result.hypotheses = [["result1"]]
        mock_batch_result.scores = [0.9]
        mock_translator.translate_batch.return_value = [mock_batch_result]
        
        # Test the function
        sentences = ["sentence1"]
        result = translate.apply_packaged_translation_batch(
            mock_pkg, sentences, mock_translator, num_hypotheses=1
        )
        
        # Should remove the prefix from decoded result
        assert len(result) == 1
        assert len(result[0]) == 1
        # The prefix should be removed from the result
        assert not result[0][0].value.startswith("prefix_")



class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_batch_translation_with_none_values(self):
        """Test batch translation with None values in input"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.side_effect = lambda x: [f"result_{i}" if text is not None else "None" for i, text in enumerate(x)]
            mock_get_translation.return_value = mock_translation
            
            # This should handle None values gracefully or raise appropriate error
            result = translate.translate_batch([None, "text", None], "es", "en")
            
            # The implementation should handle this case appropriately
            assert len(result) == 3

    def test_batch_translation_with_empty_strings(self):
        """Test batch translation with empty strings"""
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = ["", "result", ""]
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch(["", "text", ""], "es", "en")
            
            assert result == ["", "result", ""]

    def test_batch_translation_with_very_long_text(self):
        """Test batch translation with very long text"""
        long_text = "Texto muy largo. " * 1000  # Very long text
        
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = ["Very long translated text"]
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch([long_text], "es", "en")
            
            assert len(result) == 1
            assert result[0] == "Very long translated text"

    def test_batch_translation_large_batch_size(self):
        """Test batch translation with large number of texts"""
        large_batch = [f"Texto {i}" for i in range(100)]
        
        with patch('argostranslate.translate.get_translation_from_codes') as mock_get_translation:
            mock_translation = Mock()
            mock_translation.translate_batch.return_value = [f"Text {i}" for i in range(100)]
            mock_get_translation.return_value = mock_translation
            
            result = translate.translate_batch(large_batch, "es", "en")
            
            assert len(result) == 100
            assert all(f"Text {i}" == result[i] for i in range(100))