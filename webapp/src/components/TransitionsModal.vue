<template>
  <div class="text-center">
    <v-dialog v-model="show">
      <v-card style="padding: 10px">
        <img :src="'data:image/png;base64, ' + imgs_bytes[img_index - 1]"/>
        <v-card-text>
          <v-container>
            <v-row>
              <v-col md="12" class="d-flex justify-center">
                {{ img_index }} of {{ imgs_bytes.length }}
              </v-col>
            </v-row>
          
            <v-row justify="center">
              <v-col class="d-flex justify-center">
                <v-slider
                  v-model="img_index"
                  class="align-center"
                  :max="max"
                  :min="min"
                  hide-details
                  step="1"
                  show-ticks="always"
                >
                </v-slider>
              </v-col>
            </v-row>
          </v-container>
        </v-card-text>
        <v-card-actions>
          <v-btn color="primary" @click="save">Save this image</v-btn>
          <v-spacer></v-spacer>
          <v-btn color="primary" @click="closeModal">Close Dialog</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
import axios from 'axios'
export default {
    props: ['showModal', 'imgs_bytes', 'zs'],
    data() {
        return {
            show: false,
            img_index: 1,
            min: 1,
            max: 1,
        }
    },
    methods: {
        closeModal(){
            this.img_index = 1
            this.$emit('close')
        },
        save(){
            let formData = new FormData();
            formData.append('z', this.zs[this.img_index - 1])
            axios({
                url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/save',
                method: 'post',
                data: formData,
                headers: { "Content-Type": "multipart/form-data" },
            }).then((response) => {
                console.log(response)
                alert('Image saved with ID: ' + response.data.id)
            }).catch((e) => {
                alert('Error')
                console.log(e)
            })
            
        },
    },
    watch: {
        showModal(newValue, oldValue) {
            this.show = newValue
        },
        imgs_bytes(newValue, oldValue) {
            this.max = newValue.length
        }
    }
}
</script>

<style scoped>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 40%;
}
</style>