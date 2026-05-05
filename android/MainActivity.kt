package com.example.escombot

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var adapter: ChatAdapter
    private val messages = mutableListOf<Message>()
    private lateinit var chatService: ChatService
    private lateinit var recyclerView: RecyclerView
    private lateinit var editText: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var layoutSuggestions: LinearLayout

    private val chatHistoryForServer = mutableListOf<HistoryItem>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.mainactivity)

        recyclerView = findViewById(R.id.recyclerView)
        editText = findViewById(R.id.editTextMessage)
        btnSend = findViewById(R.id.btnSend)
        layoutSuggestions = findViewById(R.id.layoutSuggestions)

        adapter = ChatAdapter(messages)

        // LayoutManager robusto que evita crash por inconsistencia de índices
        val layoutManager = object : LinearLayoutManager(this) {
            override fun onLayoutChildren(recycler: RecyclerView.Recycler?, state: RecyclerView.State?) {
                try {
                    super.onLayoutChildren(recycler, state)
                } catch (e: IndexOutOfBoundsException) {
                    android.util.Log.e("ESCOMbot", "Inconsistencia de RecyclerView evitada")
                }
            }
        }
        layoutManager.stackFromEnd = true
        recyclerView.layoutManager = layoutManager
        recyclerView.adapter = adapter

        // OkHttp con timeouts generosos para el cold start de Render
        val client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(150, TimeUnit.SECONDS)  // 2.5 min — el backend espera hasta 2 min
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl("https://escombot.onrender.com/")
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        chatService = retrofit.create(ChatService::class.java)

        loadInitialSuggestions()

        btnSend.setOnClickListener {
            val text = editText.text.toString().trim()
            if (text.isNotEmpty()) {
                sendMessage(text)
            }
        }
    }

    // Verifica /health antes de enviar — evita el "se está iniciando" infinito
    private fun waitForServerAndSend(text: String, retries: Int = 0) {
        // Máximo ~60s de polling (20 intentos x 3s)
        if (retries > 20) {
            removeTypingIndicator()
            messages.add(Message("El servidor tardó demasiado en iniciar. Intenta de nuevo en un momento 😔", isBot = true))
            adapter.notifyItemInserted(messages.size - 1)
            recyclerView.scrollToPosition(messages.size - 1)
            btnSend.isEnabled = true
            return
        }

        chatService.health().enqueue(object : Callback<HealthResponse> {
            override fun onResponse(call: Call<HealthResponse>, response: Response<HealthResponse>) {
                if (response.body()?.chainLoaded == true) {
                    // Servidor listo — enviar la pregunta real
                    sendMessageToServer(text)
                } else {
                    // Todavía cargando — reintentar en 3s
                    Handler(Looper.getMainLooper()).postDelayed({
                        waitForServerAndSend(text, retries + 1)
                    }, 3000)
                }
            }

            override fun onFailure(call: Call<HealthResponse>, t: Throwable) {
                // Sin respuesta aún — reintentar en 3s
                Handler(Looper.getMainLooper()).postDelayed({
                    waitForServerAndSend(text, retries + 1)
                }, 3000)
            }
        })
    }

    private fun sendMessage(text: String) {
        layoutSuggestions.visibility = View.GONE
        btnSend.isEnabled = false

        // Mensaje del usuario
        messages.add(Message(text, isBot = false))
        adapter.notifyItemInserted(messages.size - 1)
        recyclerView.scrollToPosition(messages.size - 1)
        editText.text.clear()

        // Typing indicator
        messages.add(Message("", isBot = true, isTyping = true))
        adapter.notifyItemInserted(messages.size - 1)
        recyclerView.scrollToPosition(messages.size - 1)

        // Primero verificar si el servidor está listo, luego enviar
        waitForServerAndSend(text)
    }

    private fun sendMessageToServer(text: String) {
        val request = ChatRequest(
            question = text,
            history = chatHistoryForServer.takeLast(5)
        )

        chatService.getResponse(request).enqueue(object : Callback<ChatResponse> {
            override fun onResponse(call: Call<ChatResponse>, response: Response<ChatResponse>) {
                btnSend.isEnabled = true
                removeTypingIndicator()

                val reply = if (response.isSuccessful)
                    response.body()?.answer ?: "Sin respuesta del servidor"
                else
                    "El servidor no respondió correctamente (${response.code()})"

                messages.add(Message(reply, isBot = true))
                chatHistoryForServer.add(HistoryItem(user = text, bot = reply))
                adapter.notifyItemInserted(messages.size - 1)
                recyclerView.scrollToPosition(messages.size - 1)
            }

            override fun onFailure(call: Call<ChatResponse>, t: Throwable) {
                btnSend.isEnabled = true
                removeTypingIndicator()

                val errorMsg = when {
                    t.message?.contains("timeout", ignoreCase = true) == true ->
                        "La respuesta tardó demasiado ⏳\nIntenta de nuevo en un momento"
                    t.message?.contains("Unable to resolve host", ignoreCase = true) == true ->
                        "Sin conexión a internet 🌐"
                    else -> "Error de conexión: ${t.message}"
                }

                messages.add(Message(errorMsg, isBot = true))
                adapter.notifyItemInserted(messages.size - 1)
                recyclerView.scrollToPosition(messages.size - 1)
            }
        })
    }

    // Elimina el typing indicator de forma segura
    private fun removeTypingIndicator() {
        val typingIndex = messages.indexOfLast { it.isTyping }
        if (typingIndex != -1) {
            messages.removeAt(typingIndex)
            adapter.notifyItemRemoved(typingIndex)
        }
    }

    private fun loadInitialSuggestions() {
        messages.add(Message("¿En qué te puedo ayudar hoy?", isBot = true))
        adapter.notifyItemInserted(messages.size - 1)

        chatService.getSuggestions().enqueue(object : Callback<SuggestionResponse> {
            override fun onResponse(call: Call<SuggestionResponse>, response: Response<SuggestionResponse>) {
                if (response.isSuccessful) {
                    val sugerencias = response.body()?.suggestions ?: emptyList()
                    displaySuggestionButtons(sugerencias)
                }
            }

            override fun onFailure(call: Call<SuggestionResponse>, t: Throwable) {
                // Si falla al cargar sugerencias, mostramos las por defecto
                displaySuggestionButtons(listOf(
                    "¿Cómo solicito una beca?",
                    "Requisitos para servicio social"
                ))
            }
        })
    }

    private fun displaySuggestionButtons(sugerencias: List<String>) {
        layoutSuggestions.removeAllViews()

        for (texto in sugerencias) {
            val btn = Button(this).apply {
                this.text = texto
                this.isAllCaps = false
                this.setTextColor(android.graphics.Color.WHITE)
                this.background = androidx.core.content.ContextCompat.getDrawable(
                    context, R.drawable.bg_bot_bubble
                )
                val params = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 8, 0, 8) }
                this.layoutParams = params
                setOnClickListener { sendMessage(texto) }
            }
            layoutSuggestions.addView(btn)
        }
    }
}
