package com.example.escombot

data class Message(
    val text: String,
    val isBot: Boolean,
    val isTyping: Boolean = false  // ← AGREGAR
)