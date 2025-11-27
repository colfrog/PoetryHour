package com.nilio.poetryhour

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    private val MODEL_NAME = "gemma3.tflite"
    private val TOKENIZER_NAME = "tokenizer.spm"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            // Theme Setup
            val darkTheme = isSystemInDarkTheme()
            val colorScheme = when {
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
                    val context = LocalContext.current
                    if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
                }
                darkTheme -> darkColorScheme()
                else -> lightColorScheme()
            }

            MaterialTheme(colorScheme = colorScheme) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // Check for Disclaimer BEFORE showing the main app content
                    DisclaimerWrapper {
                        PoetryEditorScreen(MODEL_NAME, TOKENIZER_NAME)
                    }
                }
            }
        }
    }
}

/**
 * A wrapper that handles the "One-Time" Legal Disclaimer check.
 */
@Composable
fun DisclaimerWrapper(content: @Composable () -> Unit) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("app_settings", Context.MODE_PRIVATE) }

    // Check if already accepted
    var showDialog by remember {
        mutableStateOf(!prefs.getBoolean("gemma_terms_accepted", false))
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { /* Force user to accept, do not dismiss on click outside */ },
            title = { Text("Gemma Terms of Use") },
            text = {
                Column {
                    Text("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms")
                    Spacer(Modifier.height(8.dp))
                    // Optional: Clickable link for better UX
                    TextButton(
                        onClick = {
                            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://ai.google.dev/gemma/terms"))
                            context.startActivity(intent)
                        },
                        contentPadding = PaddingValues(0.dp)
                    ) {
                        Text("Read Terms Online")
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        // Save the acceptance flag
                        prefs.edit().putBoolean("gemma_terms_accepted", true).apply()
                        showDialog = false
                    }
                ) {
                    Text("I Understand")
                }
            }
        )
    } else {
        // If accepted, show the actual app
        content()
    }
}

@Composable
fun PoetryEditorScreen(modelName: String, tokenizerName: String) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Logic State
    var gemmaClient by remember { mutableStateOf<GemmaClient?>(null) }
    var status by remember { mutableStateOf("Initializing Model...") }
    var isModelReady by remember { mutableStateOf(false) }

    // Editor State
    var prompt by remember { mutableStateOf(TextFieldValue("The quick brown fox")) }
    var suggestions by remember { mutableStateOf<List<String>>(emptyList()) }
    var temperature by remember { mutableFloatStateOf(1.0f) }
    var isThinking by remember { mutableStateOf(false) }

    // --- 1. INITIALIZATION ---
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                NativeTokenizer.init(context, tokenizerName)
                gemmaClient = GemmaClient(context, modelName)
                isModelReady = true
                status = "Ready"
            } catch (e: Exception) {
                status = "Load Error: ${e.message}"
            }
        }
    }

    // --- 2. UI LAYOUT ---
    Column(
        modifier = Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .padding(16.dp)
    ) {
        // Header
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "PoetryHour",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.weight(1f))
            if (!isModelReady) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
            }
        }
        Text(text=status, style=MaterialTheme.typography.bodySmall , color=MaterialTheme.colorScheme.onBackground)

        Spacer(Modifier.height(24.dp))

        // The Editor
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Start writing...") },
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            textStyle = MaterialTheme.typography.bodyLarge,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = MaterialTheme.colorScheme.primary,
                unfocusedBorderColor = MaterialTheme.colorScheme.outline
            )
        )

        Spacer(Modifier.height(16.dp))

        // Controls
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .imePadding() // Handle Keyboard
        ) {
            // Temperature
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Creativity: ${String.format("%.1f", temperature)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            Slider(
                value = temperature,
                onValueChange = { temperature = it },
                valueRange = 0.1f..2.0f,
                steps = 19,
                modifier = Modifier.height(20.dp)
            )

            Spacer(Modifier.height(8.dp))

            // Suggest Button
            Button(
                onClick = {
                    isThinking = true
                    suggestions = emptyList()
                    scope.launch(Dispatchers.Default) {
                        val newWords = gemmaClient?.predictNextPossibleWords(
                            text = prompt.text,
                            topK = 10,
                            temperature = temperature
                        ) ?: emptyList()

                        withContext(Dispatchers.Main) {
                            suggestions = newWords
                            isThinking = false
                        }
                    }
                },
                enabled = isModelReady && !isThinking,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (isThinking) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = MaterialTheme.colorScheme.onPrimary,
                        strokeWidth = 2.dp
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Analyzing...")
                } else {
                    Text("Suggest Next Words")
                }
            }

            // Chips
            if (suggestions.isNotEmpty()) {
                Spacer(Modifier.height(16.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(suggestions) { word ->
                        SuggestionChip(
                            onClick = {
                                val currentText = prompt.text
                                val newText = "$currentText$word"
                                prompt = TextFieldValue(newText, TextRange(newText.length))
                                suggestions = emptyList()
                            },
                            label = { Text(word) }
                        )
                    }
                }
            }
        }
    }
}